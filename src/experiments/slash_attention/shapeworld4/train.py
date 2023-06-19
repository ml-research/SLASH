print("start importing...")

import time
import sys
import os
import json

wb = True
import wandb
sys.path.append('../../../')
sys.path.append('../../../SLASH/')
sys.path.append('../../../EinsumNetworks/src/')

#torch, numpy, ...
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

#own modules
from dataGen import SHAPEWORLD4
from einsum_wrapper import EiNet
from slash import SLASH
import utils
import ap_utils
from utils import set_manual_seed
from slot_attention_module import SlotAttention_model
from pathlib import Path

from rtpt import RTPT
print("...done")



def slash_slot_attention(exp_name , exp_dict):

    program ="""
    slot(s1).
    slot(s2).
    slot(s3).
    slot(s4).
    name(o1).
    name(o2).
    name(o3).
    name(o4).

    %assign each name a slot
    %build the object ontop of the slot assignment
    object(N,C,S,H,Z) :- color(0, +X, -C), shape(0, +X, -S), shade(0, +X, -H), size(0, +X, -Z), slot(X), name(N), slot_name_comb(N,X).

    npp(shade(1,X),[bright, dark, bg]) :- slot(X).
    npp(color(1,X),[red, blue, green, gray, brown, magenta, cyan, yellow, black]) :- slot(X).
    npp(shape(1,X),[circle, triangle, square, bg]) :- slot(X).
    npp(size(1,X),[small, big, bg]) :- slot(X).
    """
    
    #solve the matching problem in the logic program
    if exp_dict['hungarian_matching'] == False:
        program += """
        %assign each name a slot
        {slot_name_comb(N,X): slot(X) }=1 :- name(N). %problem we have dublicated slots
        %remove each model which has multiple slots asigned to the same name
        %:-  slot_name_comb(N1,X1), slot_name_comb(N2,X2), X1 == X2, N1 != N2.
        %remove the duplicates
        {slot_name_comb(N,X): name(N) }=1 :- slot(X). 
        """

    # Set the seeds for PRNG
    set_manual_seed(exp_dict['seed'])
    
    # Create RTPT object
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name='SLASH Attention Shapeworld4', max_iterations=int(exp_dict['epochs']))

    # Start the RTPT tracking
    rtpt.start()

    # create save paths and tensorboard writer
    writer = SummaryWriter(os.path.join("runs", exp_name,exp_dict['method'], str(exp_dict['seed'])), purge_step=0)
    saveModelPath = 'data/'+exp_name+'_'+exp_dict['method']+'/slash_slot_models_seed'+str(exp_dict['seed'])+'.pt'
    Path("data/"+exp_name+'_'+exp_dict['method']+"/").mkdir(parents=True, exist_ok=True)


    if wb:
        #start a new wandb run to track this script
        wandb.init(
        # set the wandb project where this run will be logged
        project="slash_attention",
        
        # track hyperparameters and run metadata
        config=exp_dict)

    # save args
    with open(os.path.join("runs", exp_name, 'args.json'), 'w') as json_file:
        json.dump(exp_dict, json_file, indent=4)

    print("Experiment parameters:", exp_dict)


    
    #NETWORKS
    if exp_dict['structure'] == 'poon-domingos':
        exp_dict['depth'] = None
        exp_dict['num_repetitions'] = None
        print("using poon-domingos")

    elif exp_dict['structure'] == 'binary-trees':
        exp_dict['pd_num_pieces'] = None
        print("using binary-trees")

    
    #color network
    color_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 32,
        class_count=9,
        pd_width=8,pd_height=4,
        use_em= False)

    #shape network
    shape_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 32,
        class_count=4,
        pd_width=8,pd_height=4,
        use_em= False)
    
    #shade network
    shade_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 32,
        class_count=3,
        pd_width=8,pd_height=4,
        use_em= False)
    
    #size network
    size_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 32,
        class_count=3,
        pd_width=8,pd_height=4,
        use_em= False)
    
    
    
    #create the Slot Attention network
    slot_net = SlotAttention_model(n_slots=4, n_iters=3, n_attr=18,
                                encoder_hidden_channels=32, attention_hidden_channels=64)
    slot_net = slot_net.to(device='cuda')
    
    

    #count trainable params
    num_trainable_params = [sum(p.numel() for p in color_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in shape_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in shade_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in size_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in slot_net.parameters() if p.requires_grad)]
    num_params = [sum(p.numel() for p in color_net.parameters()), 
                  sum(p.numel() for p in shape_net.parameters()),
                  sum(p.numel() for p in shade_net.parameters()),
                  sum(p.numel() for p in size_net.parameters()),
                  sum(p.numel() for p in slot_net.parameters())]
    
    print("training with {}({}) trainable params and {}({}) params in total".format(np.sum(num_trainable_params),num_trainable_params,np.sum(num_params),num_params))
    
        
    slot_net_params = list(slot_net.parameters())
    csss_params = list(color_net.parameters()) + list(shape_net.parameters()) + list(shade_net.parameters()) + list(size_net.parameters())
    
    #create the SLASH Program
    nnMapping = {'color': color_net,
                 'shape':shape_net,
                 'shade':shade_net,
                 'size':size_net}
    
    

    #OPTIMIZERS
    optimizers = {'csss': torch.optim.Adam([
                                            {'params':csss_params}],
                                            lr=exp_dict['lr'], eps=1e-7),
                 'slot': torch.optim.Adam([
                                            {'params': slot_net_params}],
                                            lr=0.0004, eps=1e-7)}

    
    SLASHobj = SLASH(program, nnMapping, optimizers)
    SLASHobj.grad_comp_device ='cpu' #set gradient computation to cpu


    print("using learning rate warmup and decay")
    warmup_epochs = exp_dict['lr_warmup_steps'] #warmup for x epochs
    decay_epochs = exp_dict['lr_decay_steps']
    slot_base_lr = 0.0004

    
    
    #metric lists
    test_ap_list = [] #stores average precsion values
    test_metric_list = [] #stores tp, fp, tn values
    lr_list = [] # store learning rate
    loss_list = []  # store training loss
    startTime = time.time()
    train_test_times = []
    sm_per_batch_list = []

    forward_time_list = []
    asp_time_list = []
    gradient_time_list = []
    backward_time_list = []
    
    print("loading data...")
    #if the hungarian matching algorithm is used we need to pass the object encodings to SLASH
    if exp_dict['hungarian_matching']:
        train_dataset_loader = torch.utils.data.DataLoader(SHAPEWORLD4('../../data/shapeworld4/',"train", ret_obj_encoding=True), shuffle=True,batch_size=exp_dict['bs'], num_workers=8)
    else:
        train_dataset_loader = torch.utils.data.DataLoader(SHAPEWORLD4('../../data/shapeworld4/',"train"), shuffle=True,batch_size=exp_dict['bs'], num_workers=8)

    test_dataset_loader = torch.utils.data.DataLoader(SHAPEWORLD4('../../data/shapeworld4/',"val", ret_obj_encoding=True), shuffle=False,batch_size=exp_dict['bs'], num_workers=8)
    obj_encodings_gt = ap_utils.get_obj_encodings(test_dataset_loader)
    print("...done")

    
    start_e= 0
    if exp_dict['resume']:
        print("resuming experiment")
        saved_model = torch.load(saveModelPath)
        
        #load pytorch models
        color_net.load_state_dict(saved_model['color_net'])
        shape_net.load_state_dict(saved_model['shape_net'])
        shade_net.load_state_dict(saved_model['shade_net'])
        size_net.load_state_dict(saved_model['size_net'])
        slot_net.load_state_dict(saved_model['slot_net'])
        
        
        #optimizers and shedulers
        optimizers['csss'].load_state_dict(saved_model['resume']['optimizer_csss'])
        optimizers['slot'].load_state_dict(saved_model['resume']['optimizer_slot'])
        start_e = saved_model['resume']['epoch']
                
        #metrics
        test_ap_list = saved_model['test_ap_list']
        test_metric_list = saved_model['test_metric_list']
        lr_list = saved_model['lr_list']
        loss_list = saved_model['loss_list']
        train_test_times = saved_model['train_test_times']
        sm_per_batch_list = saved_model['sm_per_batch_list']




    for e in range(start_e, exp_dict['epochs']):
                     
        #TRAIN
        print('Epoch {}/{}...'.format(e+1, exp_dict['epochs']))
        time_train= time.time()
        
        #apply lr schedulers to the SAm
        if e < warmup_epochs:
            lr = slot_base_lr * ((e+1)/warmup_epochs)
        else:
            lr = slot_base_lr
        lr = lr * 0.5**((e+1)/decay_epochs)
        optimizers['slot'].param_groups[0]['lr'] = lr
        lr_list.append([lr,e])

        print("LR SAm:", "{:.6f}".format(lr), optimizers['slot'].param_groups[0]['lr'])
        
        #SLASH training
        loss, forward_time, asp_time, gradient_time, backward_time, sm_per_batch= SLASHobj.learn(dataset_loader = train_dataset_loader, slot_net=slot_net, hungarian_matching=exp_dict['hungarian_matching'], method=exp_dict['method'], p_num = exp_dict['p_num'], k_num=1, batched_pass = True,
                        epoch=e, writer = writer)
        
        forward_time_list.append(forward_time)
        asp_time_list.append(asp_time)
        gradient_time_list.append(gradient_time)
        backward_time_list.append(backward_time)

        sm_per_batch_list.append(sm_per_batch)
        loss_list.append(loss)
        writer.add_scalar("train/loss", loss, global_step=e)

        timestamp_train, time_train_sec = utils.time_delta_now(time_train)
        
        
        #TEST
        time_test = time.time()

        #forward test batch
        inference = SLASHobj.forward_slot_attention_pipeline(slot_net=slot_net, dataset_loader= test_dataset_loader)                          
                            
        #compute the average precision, tp, fp, tn for color+shape+shade+size, color, shape, shade, size
        #color+shape+shade+size
        pred = ap_utils.inference_map_to_array(inference).cpu().numpy()
        ap, true_positives,false_positives, true_negatives, correctly_classified = ap_utils.average_precision(pred, obj_encodings_gt,-1, "SHAPEWORLD4")
        print("avg precision ",ap, "tp", true_positives, "fp", false_positives, "tn", true_negatives, "correctly classified",correctly_classified)
            
        
        #color
        pred_c = ap_utils.inference_map_to_array(inference, only_color=True).cpu().numpy()
        ap_c, true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c= ap_utils.average_precision(pred, obj_encodings_gt,-1, "SHAPEWORLD4", only_color = True)
        print("avg precision color",ap_c, "tp", true_positives_c, "fp", false_positives_c, "tn", true_negatives_c, "correctly classified",correctly_classified_c)

        #shape              
        pred_s = ap_utils.inference_map_to_array(inference, only_shape=True).cpu().numpy()
        ap_s, true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s= ap_utils.average_precision(pred_s, obj_encodings_gt,-1, "SHAPEWORLD4", only_shape = True)
        print("avg precision shape",ap_s, "tp", true_positives_s, "fp", false_positives_s, "tn", true_negatives_s, "correctly classified",correctly_classified_s)
        
        #shade              
        pred_h = ap_utils.inference_map_to_array(inference, only_shade=True).cpu().numpy()
        ap_h, true_positives_h, false_positives_h, true_negatives_h, correctly_classified_h= ap_utils.average_precision(pred_h, obj_encodings_gt,-1, "SHAPEWORLD4", only_shade = True)
        print("avg precision shade",ap_h, "tp", true_positives_h, "fp", false_positives_h, "tn", true_negatives_h, "correctly classified",correctly_classified_h)
        
        #shape              
        pred_x = ap_utils.inference_map_to_array(inference, only_size=True).cpu().numpy()
        ap_x, true_positives_x, false_positives_x, true_negatives_x, correctly_classified_x= ap_utils.average_precision(pred_x, obj_encodings_gt,-1, "SHAPEWORLD4", only_size = True)
        print("avg precision size",ap_x, "tp", true_positives_x, "fp", false_positives_x, "tn", true_negatives_x, "correctly classified",correctly_classified_x)

        
        #store ap, tp, fp, tn
        test_ap_list.append([ap, ap_c, ap_s, ap_h, ap_x, e])                    
        test_metric_list.append([true_positives, false_positives, true_negatives,correctly_classified,
                                 true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c,
                                 true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s,
                                 true_positives_h, false_positives_h, true_negatives_h, correctly_classified_h,
                                 true_positives_x, false_positives_x, true_negatives_x, correctly_classified_x])

        #Tensorboard outputs
        writer.add_scalar("test/ap", ap, global_step=e)
        writer.add_scalar("test/ap_c", ap_c, global_step=e)
        writer.add_scalar("test/ap_s", ap_s, global_step=e)
        writer.add_scalar("test/ap_h", ap_h, global_step=e)
        writer.add_scalar("test/ap_x", ap_x, global_step=e)

        if wb:
            wandb.log({"ap": ap,
                        "ap_c":ap_c,
                        "ap_s":ap_s,
                        "ap_h":ap_h,
                        "ap_x":ap_x,
                        "loss":loss,
                        "forward_time":forward_time,
                        "asp_time":asp_time,
                        "gradient_time": gradient_time,
                        "backward_time": backward_time,
                        "sm_per_batch": sm_per_batch})

        
        #Time measurements
        timestamp_test,time_test_sec = utils.time_delta_now(time_test)
        timestamp_total,time_total_sec =  utils.time_delta_now(startTime)
        
        train_test_times.append([time_train_sec, time_test_sec, time_total_sec])
        train_test_times_np = np.array(train_test_times)
        print('--- train time:  ---', timestamp_train, '--- total: ---',train_test_times_np[:,0].sum())
        print('--- test time:  ---' , timestamp_test, '--- total: ---',train_test_times_np[:,1].sum())
        print('--- total time from beginning:  ---', timestamp_total )
        
        #save the neural network  such that we can use it later
        print('Storing the trained model into {}'.format(saveModelPath))
        torch.save({"shape_net":  shape_net.state_dict(), 
                    "color_net": color_net.state_dict(),
                    "shade_net": shade_net.state_dict(),
                    "size_net": size_net.state_dict(),
                    "slot_net": slot_net.state_dict(),
                    "resume": {
                        "optimizer_csss":optimizers['csss'].state_dict(),
                        "optimizer_slot": optimizers['slot'].state_dict(),
                        "epoch":e
                    },
                    "test_ap_list":test_ap_list,
                    "loss_list":loss_list,
                    "sm_per_batch_list":sm_per_batch_list,
                    "test_metric_list":test_metric_list,
                    "lr_list": lr_list,
                    "num_params": num_params,
                    "train_test_times": train_test_times,
                    "exp_dict":exp_dict,
                    "time": {
                        "forward_time":forward_time_list,
                        "asp_time": asp_time_list,
                        "gradient_time":gradient_time_list,
                        "backward_time":backward_time_list,
                    },
                    "program":program}, saveModelPath)
        
        # Update the RTPT
        rtpt.step(subtitle=f"ap={ap:2.2f}")
    if wb:
        wandb.finish()
        



