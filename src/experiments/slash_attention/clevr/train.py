print("start importing...")

import time
import sys
import os
import json
sys.path.append('../../../')
sys.path.append('../../../SLASH/')
sys.path.append('../../../EinsumNetworks/src/')

wb = True
import wandb

#torch, numpy, ...
import torch
from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()


import numpy as np
import importlib

#own modules

from dataGen import CLEVR
from auxiliary import get_files_names_and_paths, get_slash_program
from einsum_wrapper import EiNet
from slash import SLASH
import utils
import ap_utils
from utils import set_manual_seed
from slot_attention_module import SlotAttention_model
from slot_attention_module import SlotAttention_model
from pathlib import Path
from rtpt import RTPT
print("...done")



def slash_slot_attention(exp_name , exp_dict):
    
    # Set the seeds for PRNG
    set_manual_seed(exp_dict['seed'])

    # Create RTPT object
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name=f'SLASH Attention CLEVR %s' % exp_dict['obj_num'], max_iterations=int(exp_dict['epochs']))
    
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

    if wb:
        #start a new wandb run to track this script
        wandb.init(
        # set the wandb project where this run will be logged
        project="slash_attention_clevr",
        
        # track hyperparameters and run metadata
        config=exp_dict)


    #setup new SLASH program given the network parameters
    program = get_slash_program(exp_dict['obj_num'])
    
    #setup new SLASH program given the network parameters
    if exp_dict['structure'] == 'poon-domingos':
        exp_dict['depth'] = None
        exp_dict['num_repetitions'] = None
        print("using poon-domingos")

    elif exp_dict['structure'] == 'binary-trees':
        exp_dict['pd_num_pieces'] = None
        print("using binary-trees")
              
    #size network
    size_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 64,
        pd_width = 8,
        pd_height = 8,
        class_count = 3,
        use_em = exp_dict['use_em'],
        learn_prior = exp_dict['learn_prior'])
    
    #material network
    material_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 64,
        pd_width = 8,
        pd_height = 8,
        class_count = 3,
        use_em = exp_dict['use_em'],
        learn_prior = exp_dict['learn_prior'])
    
    #shape network
    shape_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 64,
        pd_width = 8,
        pd_height = 8,
        class_count = 4,
        use_em = exp_dict['use_em'],
        learn_prior = exp_dict['learn_prior'])
    
    #color network
    color_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 64,
        pd_width = 8,
        pd_height = 8,
        class_count = 9,
        use_em = exp_dict['use_em'],
        learn_prior = exp_dict['learn_prior'])
        
    
    
    #create the Slot Attention network
    slot_net = SlotAttention_model(n_slots=exp_dict['obj_num'], n_iters=3, n_attr=18,
                                   encoder_hidden_channels=64, attention_hidden_channels=128, clevr_encoding=True)
    slot_net = slot_net.to(device='cuda')
        


    #trainable params
    num_trainable_params = [sum(p.numel() for p in size_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in material_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in shape_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in color_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in slot_net.parameters() if p.requires_grad)]
    num_params = [sum(p.numel() for p in size_net.parameters()), 
                  sum(p.numel() for p in material_net.parameters()), 
                  sum(p.numel() for p in shape_net.parameters()), 
                  sum(p.numel() for p in color_net.parameters()),
                  sum(p.numel() for p in slot_net.parameters())]
    
    print("training with {}({}) trainable params and {}({}) params in total".format(np.sum(num_trainable_params),num_trainable_params,np.sum(num_params),num_params))
         
            
    slot_net_params = list(slot_net.parameters())
    smsc_params = list(size_net.parameters()) + list(material_net.parameters()) + list(shape_net.parameters())  + list(color_net.parameters()) 
    
    #create the SLASH Program
    nnMapping = {'size': size_net,
                 'material': material_net,
                 'shape': shape_net,
                 'color': color_net
                }
        
    
    #OPTIMIZERS
    optimizers = {'smsc': torch.optim.Adam([
                                        {'params':smsc_params}],
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
    
    # Load data
    obj_num = exp_dict['obj_num']
    root = '/SLASH/data/CLEVR_v1.0/'
    mode = 'train'
    img_paths, files_names = get_files_names_and_paths(root=root, mode=mode, obj_num=obj_num)
    
    train_dataset_loader = torch.utils.data.DataLoader(CLEVR(root,mode,img_paths,files_names,obj_num), shuffle=True,batch_size=exp_dict['bs'],num_workers=8)
    
    mode = 'val'
    img_paths, files_names = get_files_names_and_paths(root=root, mode=mode, obj_num=obj_num)
    test_dataset_loader= torch.utils.data.DataLoader(CLEVR(root,mode,img_paths,files_names,obj_num), shuffle=False,batch_size=exp_dict['bs'], num_workers=8)

    obj_encodings_gt = ap_utils.get_obj_encodings(test_dataset_loader)

    print("loaded data")
    
    
    # Resume the training if requested
    start_e= 0
    if exp_dict['resume']:
        print("resuming experiment")
        saved_model = torch.load(saveModelPath)
        
        #load pytorch models
        color_net.load_state_dict(saved_model['color_net'])
        shape_net.load_state_dict(saved_model['shape_net'])
        material_net.load_state_dict(saved_model['material_net'])
        size_net.load_state_dict(saved_model['size_net'])
        slot_net.load_state_dict(saved_model['slot_net'])
        
        
        #optimizers and shedulers
        optimizers['smsc'].load_state_dict(saved_model['resume']['optimizer_smsc'])
        optimizers['slot'].load_state_dict(saved_model['resume']['optimizer_slot'])
        start_e = saved_model['resume']['epoch']
    
        #metrics
        test_ap_list = saved_model['test_ap_list']
        test_metric_list = saved_model['test_metric_list']
        lr_list = saved_model['lr_list']
        loss_list = saved_model['loss_list']
        train_test_times = saved_model['train_test_times']
        sm_per_batch_list = saved_model['sm_per_batch_list']  
    

    

    # train the network and evaluate the performance
    for e in range(start_e, exp_dict['epochs']):
        #we have three datasets right now train, val and test with 20k, 5k and 100 samples
                
        #TRAIN
        print('Epoch {}/{}...'.format(e+1, exp_dict['epochs']))
        time_train = time.time()
        
        #apply lr schedulers
        if e < warmup_epochs:
            lr = slot_base_lr * ((e+1)/warmup_epochs)
        else:
            lr = slot_base_lr
        lr = lr * 0.5**((e+1)/decay_epochs)
        optimizers['slot'].param_groups[0]['lr'] = lr
        lr_list.append([lr,e])
        print("LR SAm:", "{:.6f}".format(lr), optimizers['slot'].param_groups[0]['lr'])


        loss, forward_time, asp_time, gradient_time, backward_time, sm_per_batch  = SLASHobj.learn(dataset_loader = train_dataset_loader, slot_net=slot_net,hungarian_matching = True, method=exp_dict['method'], p_num=exp_dict['p_num'], k_num=1,
                              epoch=e, writer = writer, batched_pass = True)
        
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

        #compute the average precision, tp, fp, tn for color+shape+material+size
        pred = ap_utils.inference_map_to_array(inference).cpu().numpy()


        #use only a fraction of the test data, except every 100th epoch
        if e % 100 == 0:
            subset_size = None
        else:
            subset_size = 1500

        ap, true_positives,false_positives, true_negatives, correctly_classified  = ap_utils.average_precision(pred, obj_encodings_gt, -1, "CLEVR",subset_size = subset_size)
        print("avg precision ", ap, "tp", true_positives, "fp", false_positives, "tn", true_negatives, "correctly classified", correctly_classified)

        '''
        #color
        pred_c = ap_utils.inference_map_to_array(inference, only_color=True).cpu().numpy()
        ap_c, true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c = ap_utils.average_precision(pred, obj_encodings_gt, -1, "CLEVR", only_color=True)
        print("avg precision color", ap_c, "tp", true_positives_c, "fp", false_positives_c, "tn", true_negatives_c, "correctly classified", correctly_classified_c)

        #shape              
        pred_s = ap_utils.inference_map_to_array(inference, only_shape=True).cpu().numpy()
        ap_s, true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s = ap_utils.average_precision(pred_s, obj_encodings_gt, -1, "CLEVR", only_shape=True)
        print("avg precision shape", ap_s, "tp", true_positives_s, "fp", false_positives_s, "tn", true_negatives_s, "correctly classified", correctly_classified_s)
        
        #material              
        pred_m = ap_utils.inference_map_to_array(inference, only_material=True).cpu().numpy()
        ap_m, true_positives_m, false_positives_m, true_negatives_m, correctly_classified_m = ap_utils.average_precision(pred_m, obj_encodings_gt, -1, "CLEVR", only_material=True)
        print("avg precision material", ap_m, "tp", true_positives_m, "fp", false_positives_m, "tn", true_negatives_m, "correctly classified", correctly_classified_m)
        
        #size              
        pred_x = ap_utils.inference_map_to_array(inference, only_size=True).cpu().numpy()
        ap_x, true_positives_x, false_positives_x, true_negatives_x, correctly_classified_x = ap_utils.average_precision(pred_x, obj_encodings_gt, -1, "CLEVR", only_size=True)
        print("avg precision size", ap_x, "tp", true_positives_x, "fp", false_positives_x, "tn", true_negatives_x, "correctly classified", correctly_classified_x)

        
        #store ap, tp, fp, tn
        test_ap_list.append([ap, ap_c, ap_s, ap_m, ap_x, e])                    
        test_metric_list.append([true_positives, false_positives, true_negatives, correctly_classified,
                                 true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c,
                                 true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s,
                                 true_positives_m, false_positives_m, true_negatives_m, correctly_classified_m,
                                 true_positives_x, false_positives_x, true_negatives_x, correctly_classified_x])
        '''

        if wb:
            wandb.log({"ap": ap,
                        "loss":loss,
                        "forward_time":np.sum(forward_time),
                        "asp_time":np.sum(asp_time),
                        "gradient_time": np.sum(gradient_time),
                        "backward_time": np.sum(backward_time),
                        "sm_per_batch": np.sum(sm_per_batch)})

        test_ap_list.append([ap, e])                    
        test_metric_list.append([true_positives, false_positives, true_negatives, correctly_classified])
        
        #Tensorboard outputs
        writer.add_scalar("test/ap", ap, global_step=e)
        #writer.add_scalar("test/ap_c", ap_c, global_step=e)
        #writer.add_scalar("test/ap_s", ap_s, global_step=e)
        #writer.add_scalar("test/ap_m", ap_m, global_step=e)
        #writer.add_scalar("test/ap_x", ap_x, global_step=e)


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
        torch.save({"color_net":color_net.state_dict(),
                    "shape_net":shape_net.state_dict(),                    
                    "material_net":material_net.state_dict(),
                    "size_net":size_net.state_dict(),
                    "slot_net":slot_net.state_dict(),
                    "resume": {
                        "optimizer_smsc":optimizers['smsc'].state_dict(),
                        "optimizer_slot":optimizers['slot'].state_dict(),
                        "epoch":e
                    },
                    "test_ap_list":test_ap_list,
                    "loss_list":loss_list,                    
                    "sm_per_batch_list":sm_per_batch_list,
                    "test_metric_list":test_metric_list,
                    "lr_list":lr_list,
                    "num_params":num_params,
                    "train_test_times": train_test_times,
                    "time": {
                        "forward_time":forward_time_list,
                        "asp_time": asp_time_list,
                        "gradient_time":gradient_time_list,
                        "backward_time":backward_time_list,
                    },
                    "exp_dict":exp_dict,
                    "program":program}, saveModelPath)
        
        # Update the RTPT
        rtpt.step()
    if wb:
        wandb.finish()
        