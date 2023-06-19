print("start importing...")

import os
import time
import sys
import json
sys.path.append('../../../')
sys.path.append('../../../SLASH/')
sys.path.append('../../../EinsumNetworks/src/')

#torch, numpy, ...
import torch
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler


import numpy as np

#own modules
from dataGen import SHAPEWORLD_COGENT
from einsum_wrapper import EiNet
from slash import SLASH
import ap_utils
import utils
from utils import set_manual_seed

from slot_attention_module import SlotAttention_model
from pathlib import Path

from rtpt import RTPT

#seeds
torch.manual_seed(42)
np.random.seed(42)
print("...done")






def slash_slot_attention(exp_name , exp_dict):
    program ='''
    slot(s1).
    slot(s2).
    slot(s3).
    slot(s4).
    name(o1).
    name(o2).
    name(o3).
    name(o4).

    %build the object ontop of the slot assignment
    object(N,C,S,H,Z) :- color(0, +X, -C), shape(0, +X, -S), shade(0, +X, -H), size(0, +X, -Z), slot(X), name(N), slot_name_comb(N,X).

    npp(color(1,X),[red, blue, green, gray, brown, magenta, cyan, yellow, black]) :- slot(X).
    npp(shape(1,X),[circle, triangle, square, bg]) :- slot(X).
    npp(shade(1,X),[bright, dark, bg]) :- slot(X).
    npp(size(1,X),[small, big, bg]) :- slot(X).
    '''

    if exp_dict['hungarian_matching'] == False:
        program += """
        %assign each name a slot
        {slot_name_comb(N,X): slot(X) }=1 :- name(N). %problem we have dublicated slots
        %remove each model which has multiple slots asigned to the same name
        %:-  slot_name_comb(N1,X1), slot_name_comb(N2,X2), X1 == X2, N1 != N2.
        {slot_name_comb(N,X): name(N) }=1 :- slot(X). %problem we have dublicated slots
        """

    # Set the seeds for PRNG
    set_manual_seed(exp_dict['seed'])
    
    # Create RTPT object
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name='SLASH Shapeworld4 CoGenT', max_iterations=int(exp_dict['epochs']))

    # Start the RTPT tracking
    rtpt.start()
    
    
    # create save paths and tensorboard writer
    writer = SummaryWriter(os.path.join("runs", exp_name,exp_dict['method'], str(exp_dict['seed'])), purge_step=0)
    saveModelPath = 'data/'+exp_name+'_'+exp_dict['method']+'/slash_slot_models_seed'+str(exp_dict['seed'])+'.pt'
    Path("data/"+exp_name+'_'+exp_dict['method']+"/").mkdir(parents=True, exist_ok=True)


    # save args
    with open(os.path.join("runs", exp_name, 'args.json'), 'w') as json_file:
        json.dump(exp_dict, json_file, indent=4)

    print("Experiment parameters:", exp_dict)


    #setup new SLASH program given the network parameters
    if exp_dict['structure'] == 'poon-domingos':
        exp_dict['depth'] = None
        exp_dict['num_repetitions'] = None
        print("using poon-domingos")

    elif exp_dict['structure'] == 'binary-trees':
        exp_dict['pd_num_pieces'] = None
        print("using binary-trees")

    
    #NETWORKS
        
    #color network
    color_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 64,
        class_count=9,
        pd_width=8,pd_height=8,
        use_em= False)

    #shape network
    shape_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 64,
        class_count=4,
        pd_width=8,pd_height=8,
        use_em= False)
    
    #shade network
    shade_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 64,
        class_count=3,
        pd_width=8,pd_height=8,
        use_em= False)
    
    #size network
    size_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 64,
        class_count=3,
        pd_width=8,pd_height=8,
        use_em= False)
    
    
    
    #create the Slot Attention network
    slot_net = SlotAttention_model(n_slots=4, n_iters=3, n_attr=18,
                                encoder_hidden_channels=64, attention_hidden_channels=64)
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
    
    

    #OPTIMIZERS and LEARNING RATE SHEDULING
    print("training probabilisitc circuits and SlotAttention module")
    optimizers = {'csss': torch.optim.Adam([
                                            {'params':csss_params}],
                                            lr=exp_dict['lr'], eps=1e-7),
                 'slot': torch.optim.Adam([
                                            {'params': slot_net_params}],
                                            lr=0.0004, eps=1e-7)}
   
    
    SLASHobj = SLASH(program, nnMapping, optimizers)
    
    print("using learning rate warmup and decay")
    warmup_epochs = exp_dict['lr_warmup_steps'] #warmup for x epochs
    decay_epochs = exp_dict['lr_decay_steps']
    slot_base_lr = 0.0004
    

    
    #metric lists
    test_ap_list_a = [] #stores average precsion values for test set with condition a
    test_ap_list_b = [] #stores average precsion values for test set with condition b

    test_metric_list_a = [] #stores tp, fp, tn values for test set with condition a
    test_metric_list_b = [] #stores tp, fp, tn values for test set with condition b

    lr_list = [] # store learning rate
    loss_list = []  # store training loss

    startTime = time.time()
    train_test_times = []
    sm_per_batch_list = []
    
    # load datasets
    print("loading data...")

    #if the hungarian matching algorithm is used we need to pass the object encodings to SLASH
    if exp_dict['hungarian_matching']:
        train_dataset_loader = torch.utils.data.DataLoader(SHAPEWORLD_COGENT('../../data/shapeworld_cogent/',"train_a", ret_obj_encoding=True), shuffle=True,batch_size=exp_dict['bs'],pin_memory=False, num_workers=4)
    else:
        train_dataset_loader = torch.utils.data.DataLoader(SHAPEWORLD_COGENT('../../data/shapeworld_cogent/',"train_a"), shuffle=True,batch_size=exp_dict['bs'],pin_memory=False, num_workers=4)
    
    test_dataset_loader_a = torch.utils.data.DataLoader(SHAPEWORLD_COGENT('../../data/shapeworld_cogent/',"val_a", ret_obj_encoding=True), shuffle=False,batch_size=exp_dict['bs'],pin_memory=False, num_workers=4)
    test_dataset_loader_b = torch.utils.data.DataLoader(SHAPEWORLD_COGENT('../../data/shapeworld_cogent/',"val_b", ret_obj_encoding=True), shuffle=False,batch_size=exp_dict['bs'],pin_memory=False, num_workers=4)


    obj_encodings_gt_a = ap_utils.get_obj_encodings(test_dataset_loader_a)
    obj_encodings_gt_b = ap_utils.get_obj_encodings(test_dataset_loader_b)
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
        test_ap_list_a = saved_model['test_ap_list_a']
        test_ap_list_b = saved_model['test_ap_list_b']

        test_metric_list_a = saved_model['test_metric_list_a']
        test_metric_list_b = saved_model['test_metric_list_b']

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
        
        loss,_,_,_,sm_per_batch = SLASHobj.learn(dataset_loader = train_dataset_loader, slot_net=slot_net, hungarian_matching=exp_dict['hungarian_matching'], method=exp_dict['method'], p_num = exp_dict['p_num'], k_num=1,
                        epoch=e, writer = writer)
        
        sm_per_batch_list.append(sm_per_batch)
        loss_list.append(loss)
        writer.add_scalar("train/loss", loss, global_step=e)

        timestamp_train, time_train_sec = utils.time_delta_now(time_train)
        
        
        #TEST
        time_test = time.time()

        ### CONDITION A
        print("condition a")

        #forward test batch a
        inference = SLASHobj.forward_slot_attention_pipeline(slot_net=slot_net, dataset_loader= test_dataset_loader_a)
                            
        #compute the average precision, tp, fp, tn for color+shape+shade+size, color, shape
        #color+shape+shade+size
        pred = ap_utils.inference_map_to_array(inference).cpu().numpy()
        ap, true_positives,false_positives, true_negatives, correctly_classified = ap_utils.average_precision(pred, obj_encodings_gt_a,-1, "SHAPEWORLD4")
        print("avg precision(a) ",ap, "tp", true_positives, "fp", false_positives, "tn", true_negatives, "correctly classified",correctly_classified)
                   
        #color
        pred_c = ap_utils.inference_map_to_array(inference, only_color=True).cpu().numpy()
        ap_c, true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c= ap_utils.average_precision(pred_c, obj_encodings_gt_a,-1, "SHAPEWORLD4", only_color = True)
        print("avg precision(a) color",ap_c, "tp", true_positives_c, "fp", false_positives_c, "tn", true_negatives_c, "correctly classified",correctly_classified_c)

        #shape              
        pred_s = ap_utils.inference_map_to_array(inference, only_shape=True).cpu().numpy()
        ap_s, true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s= ap_utils.average_precision(pred_s, obj_encodings_gt_a,-1, "SHAPEWORLD4", only_shape = True)
        print("avg precision(a) shape",ap_s, "tp", true_positives_s, "fp", false_positives_s, "tn", true_negatives_s, "correctly classified",correctly_classified_s)
        
    
        #store ap, tp, fp, tn
        test_ap_list_a.append([ap, ap_c, ap_s, e])                    
        test_metric_list_a.append([true_positives, false_positives, true_negatives,correctly_classified,
                                 true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c,
                                 true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s])

        
        writer.add_scalar("test/ap_cond_a", ap, global_step=e)
        writer.add_scalar("test/ap_c_cond_a", ap_c, global_step=e)
        writer.add_scalar("test/ap_s_cond_a", ap_s, global_step=e)

    
    
        ### CONDITION B
        print("condition b")
         #forward test batch a
        inference = SLASHobj.forward_slot_attention_pipeline(slot_net=slot_net, dataset_loader= test_dataset_loader_b)
             
                            
        #compute the average precision, tp, fp, tn for color+shape+shade+size, color, shap
        #color+shape+shade+size
        pred = ap_utils.inference_map_to_array(inference).cpu().numpy()
        ap, true_positives,false_positives, true_negatives, correctly_classified = ap_utils.average_precision(pred, obj_encodings_gt_b,-1, "SHAPEWORLD4")
        print("avg precision(b) ",ap, "tp", true_positives, "fp", false_positives, "tn", true_negatives, "correctly classified",correctly_classified)
                   
        #color
        pred_c = ap_utils.inference_map_to_array(inference, only_color=True).cpu().numpy()
        ap_c, true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c= ap_utils.average_precision(pred, obj_encodings_gt_b,-1, "SHAPEWORLD4", only_color = True)
        print("avg precision(b) color",ap_c, "tp", true_positives_c, "fp", false_positives_c, "tn", true_negatives_c, "correctly classified",correctly_classified_c)

        #shape              
        pred_s = ap_utils.inference_map_to_array(inference, only_shape=True).cpu().numpy()
        ap_s, true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s= ap_utils.average_precision(pred_s, obj_encodings_gt_b,-1, "SHAPEWORLD4", only_shape = True)
        print("avg precision(b) shape",ap_s, "tp", true_positives_s, "fp", false_positives_s, "tn", true_negatives_s, "correctly classified",correctly_classified_s)
            
        
        #store ap, tp, fp, tn
        test_ap_list_b.append([ap, ap_c, ap_s, e])                    
        test_metric_list_b.append([true_positives, false_positives, true_negatives,correctly_classified,
                                 true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c,
                                 true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s])
                            
        writer.add_scalar("test/ap_cond_b", ap, global_step=e)
        writer.add_scalar("test/ap_c_cond_b", ap_c, global_step=e)
        writer.add_scalar("test/ap_s_cond_b", ap_s, global_step=e)

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
                    "test_ap_list_a":test_ap_list_a,
                    "test_ap_list_b":test_ap_list_b,
                    "loss_list":loss_list,
                    "sm_per_batch_list":sm_per_batch_list,
                    "test_metric_list_a":test_metric_list_a,
                    "test_metric_list_b":test_metric_list_b,
                    "lr_list": lr_list,
                    "num_params": num_params,
                    "train_test_times": train_test_times,
                    "exp_dict":exp_dict,
                    "program":program}, saveModelPath)
        
        # Update the RTPT
        rtpt.step()

        



