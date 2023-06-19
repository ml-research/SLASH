print("start importing...")

import time
import sys
import os
import json
sys.path.append('../../../')
sys.path.append('../../../SLASH/')
sys.path.append('../../../EinsumNetworks/src/')

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
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name=f'SLASH Attention CLEVR-Cogent %s' % exp_dict['obj_num'], max_iterations=int(exp_dict['epochs']))
    
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

    

    print("using learning rate warmup and decay")
    warmup_epochs = exp_dict['lr_warmup_steps'] #warmup for x epochs
    decay_epochs = exp_dict['lr_decay_steps']
    slot_base_lr = 0.0004
        
        
    #metric lists
    test_ap_list_a = [] #stores average precsion values
    test_ap_list_b = [] #stores average precsion values

    test_metric_list_a = [] #stores tp, fp, tn values
    test_metric_list_b = [] #stores tp, fp, tn values

    lr_list = [] # store learning rate
    loss_list = []  # store training loss
    
    startTime = time.time()
    train_test_times = []
    sm_per_batch_list = []
    
    # Load data
    obj_num = exp_dict['obj_num']
    root = '/SLASH/data/CLEVR_CoGenT_v1.0/'
    
    mode = 'trainA'
    img_paths, files_names = get_files_names_and_paths(root=root, mode=mode, obj_num=obj_num)
    train_dataset_loader = torch.utils.data.DataLoader(CLEVR(root,mode,img_paths,files_names,obj_num), shuffle=True,batch_size=exp_dict['bs'],pin_memory=True, num_workers=4)
    
    mode = 'valA'
    img_paths, files_names = get_files_names_and_paths(root=root, mode=mode, obj_num=obj_num)
    test_dataset_loader_a= torch.utils.data.DataLoader(CLEVR(root,mode,img_paths,files_names,obj_num), shuffle=False,batch_size=exp_dict['bs'],pin_memory=True, num_workers=4)
    obj_encodings_gt_a = ap_utils.get_obj_encodings(test_dataset_loader_a)

    mode = 'valB'
    img_paths, files_names = get_files_names_and_paths(root=root, mode=mode, obj_num=obj_num)
    test_dataset_loader_b= torch.utils.data.DataLoader(CLEVR(root,mode,img_paths,files_names,obj_num), shuffle=False,batch_size=exp_dict['bs'],pin_memory=True, num_workers=4)
    obj_encodings_gt_b = ap_utils.get_obj_encodings(test_dataset_loader_b)

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
        test_ap_list_a = saved_model['test_ap_list_a']
        test_ap_list_b = saved_model['test_ap_list_b']

        test_metric_list_a = saved_model['test_metric_list_a']
        test_metric_list_b = saved_model['test_metric_list_b']

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


        loss,_,_,_,sm_per_batch  = SLASHobj.learn(dataset_loader = train_dataset_loader, slot_net=slot_net,hungarian_matching = True, method=exp_dict['method'], p_num=exp_dict['p_num'], k_num=1,
                              epoch=e, writer = writer)
        

        sm_per_batch_list.append(sm_per_batch)
        loss_list.append(loss)
        writer.add_scalar("train/loss", loss, global_step=e)

        
        timestamp_train, time_train_sec = utils.time_delta_now(time_train)
                
        #TEST
        time_test = time.time()

        #use only a fraction of the test data, except every 100th epoch
        if e % 100 == 0:
            subset_size = None
        else:
            subset_size = 1500

        ### CONDITION A
        print("condition a")
        
        #forward test batch
        inference = SLASHobj.forward_slot_attention_pipeline(slot_net=slot_net, dataset_loader= test_dataset_loader_a)

        #compute the average precision, tp, fp, tn for color+shape+material+size
        pred = ap_utils.inference_map_to_array(inference).cpu().numpy()
        ap_a, true_positives_a,false_positives_a, true_negatives_a, correctly_classified_a  = ap_utils.average_precision(pred, obj_encodings_gt_a, -1, "CLEVR",subset_size = subset_size)
        print("avg precision A ", ap_a, "tp", true_positives_a, "fp", false_positives_a, "tn", true_negatives_a, "correctly classified", correctly_classified_a)

        test_ap_list_a.append([ap_a, e])                    
        test_metric_list_a.append([true_positives_a, false_positives_a, true_negatives_a, correctly_classified_a])
        
        ### CONDITION B
        print("condition b")
        
        #forward test batch
        inference = SLASHobj.forward_slot_attention_pipeline(slot_net=slot_net, dataset_loader= test_dataset_loader_b)

        #compute the average precision, tp, fp, tn for color+shape+material+size
        pred = ap_utils.inference_map_to_array(inference).cpu().numpy()
        ap_b, true_positives_b,false_positives_b, true_negatives_b, correctly_classified_b  = ap_utils.average_precision(pred, obj_encodings_gt_b, -1, "CLEVR",subset_size = subset_size)
        print("avg precision A ", ap_b, "tp", true_positives_b, "fp", false_positives_b, "tn", true_negatives_b, "correctly classified", correctly_classified_b)

        test_ap_list_b.append([ap_b, e])                    
        test_metric_list_b.append([true_positives_b, false_positives_b, true_negatives_b, correctly_classified_b])
        
        
        #Tensorboard outputs
        writer.add_scalar("test/ap_a", ap_a, global_step=e)
        writer.add_scalar("test/ap_b", ap_b, global_step=e)


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
                    "test_ap_list_a":test_ap_list_a,
                    "test_ap_list_b":test_ap_list_b,
                    "loss_list":loss_list,                    
                    "sm_per_batch_list":sm_per_batch_list,
                    "test_metric_list_a":test_metric_list_a,
                    "test_metric_list_b":test_metric_list_b,
                    "lr_list":lr_list,
                    "num_params":num_params,
                    "train_test_times": train_test_times,
                    "exp_dict":exp_dict,
                    "program":program}, saveModelPath)
        
        # Update the RTPT
        rtpt.step(subtitle=f"ap_b={ap_b:2.2f}")
        