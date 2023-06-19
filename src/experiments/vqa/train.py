print("start importing...")

import time
import sys
import argparse
import datetime

sys.path.append('../../')
sys.path.append('../../SLASH/')
sys.path.append('../../EinsumNetworks/src/')


#torch, numpy, ...
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import torchvision

import numpy as np

import json

#own modules
from dataGen import VQA
from einsum_wrapper import EiNet
from network_nn import Net_nn

from tqdm import tqdm

#import slash
from slash import SLASH
import os
from datetime import date




import utils
from utils import set_manual_seed
from pathlib import Path
from rtpt import RTPT
import pickle


from knowledge_graph import RULES, KG
from dataGen import name_npp, relation_npp, attribute_npp
from models import name_clf, rela_clf, attr_clf, rela_einet, name_einet, attr_einet

print("...done")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate of model"
    )
    parser.add_argument(
        "--network-type",
        choices=["nn","pc"],
        help="The type of external to be used e.g. neural net or probabilistic circuit",
    )
    parser.add_argument(
        "--pc-structure",
        choices=["poon-domingos","binary-trees"],
        help="The type of external to be used e.g. neural net or probabilistic circuit",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=6, help="Number of threads for data loader"
    )

    parser.add_argument(
        "--p-num", type=int, default=8, help="Number of processes to devide the batch for parallel processing"
    )

    parser.add_argument(
        "--exp-name", type=str, default="vqa", help="Name of the experiment"
    )

    parser.add_argument("--credentials", type=str, help="Credentials for rtpt")


    args = parser.parse_args()

    if args.network_type == 'pc':
        args.use_pc = True
    else:
        args.use_pc = False

    return args


def determine_max_objects(task_file):

    with open (task_file, 'rb') as tf:
        tasks = pickle.load(tf)
        print("taskfile len",len(tasks))

    #get the biggest number of objects in the image
    max_objects = 0
    for tidx, task in enumerate(tasks):
        all_oid = task['question']['input']
        len_all_oid = len(all_oid)

        #store the biggest number of objects in an image
        if len_all_oid > max_objects:
            max_objects = len_all_oid
    return max_objects


 

def slash_vqa():

    args = get_args()
    print(args)

    # Set the seeds for PRNG
    set_manual_seed(args.seed)

    # Create RTPT object
    rtpt = RTPT(name_initials=args.credentials, experiment_name='SLASH VQA', max_iterations=args.epochs)

    # Start the RTPT tracking
    rtpt.start()
    exp_name = args.exp_name+ "_"+date.today().strftime("%d_%m_%Y")

    writer = SummaryWriter(os.path.join("runs", exp_name, str(args.seed)), purge_step=0)
    print(exp_name)

    Path("data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)

    #TODO workaround that adds +- notation 
    program_example = """
%scallop conversion rules
name(O,N) :-  name(0,+O,-N).
attr(O,A) :-  attr(0, +O, -A).
relation(O1,O2,N) :-  relation(0, +(O1,O2), -N).
    """

    # datasets
    #train_f = "dataset/task_list/train_tasks.pkl"  # Training dataset
    train_f = "dataset/task_list/train_tasks_c2_10000.pkl"
    val_f = "dataset/task_list/val_tasks.pkl"  # Test datset
    test_f = "dataset/task_list/test_tasks.pkl"  # Test datset
    # test_f = {"c2":"dataset/task_list/test_tasks_c2_1000.pkl",
    #           "c3":"dataset/task_list/test_tasks_c3_1000.pkl",
    #           "c4":"dataset/task_list/test_tasks_c4_1000.pkl",
    #           "c5":"dataset/task_list/test_tasks_c5_1000.pkl",
    #           "c6":"dataset/task_list/test_tasks_c6_1000.pkl"}
    

    vqa_params = {"l":100,
                "l_split":5,
                "num_names":500,
                "max_models":10000,
                "asp_timeout": 30 }

    num_obj = []
    num_obj.append(determine_max_objects(train_f))
    num_obj.append(determine_max_objects(val_f))

    if type(test_f) == str: 
        num_obj.append(determine_max_objects(test_f))
    #if we have multiple test files
    elif type(test_f) == dict:
        for key in test_f:   
            num_obj.append(determine_max_objects(test_f[key]))


    NUM_OBJECTS = np.max(num_obj)
    NUM_OBJECTS = 70

    train_data = VQA("train", train_f, NUM_OBJECTS)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_data = VQA("val", val_f, NUM_OBJECTS)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if type(test_f) == str: 
        test_data = VQA("test", test_f, NUM_OBJECTS)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    #if we have multiple test files
    elif type(test_f) == dict: 
        for key in test_f:
            test_data = VQA("test", test_f[key], NUM_OBJECTS)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
            test_f[key] = test_loader
     
    #create the SLASH Program
    # nnMapping = {'relation': rela_einet, 'name':name_einet , "attr":attr_einet}
    # optimizers = {'relation': torch.optim.Adam(rela_einet.parameters(), lr=args.lr, eps=1e-7),
    #                   'name': torch.optim.Adam(name_einet.parameters(), lr=args.lr, eps=1e-7),
    #                   'attr': torch.optim.Adam(attr_einet.parameters(), lr=args.lr, eps=1e-7)}

    nnMapping = {'relation':rela_clf, 'name':name_clf , "attr":attr_clf}
    optimizers = {'relation': torch.optim.Adam(rela_clf.parameters(), lr=args.lr, eps=1e-7),
                      'name': torch.optim.Adam(name_clf.parameters(), lr=args.lr, eps=1e-7),
                      'attr': torch.optim.Adam(attr_clf.parameters(), lr=args.lr, eps=1e-7)}

    num_trainable_params = [sum(p.numel() for p in rela_clf.parameters() if p.requires_grad),
                            sum(p.numel() for p in name_clf.parameters() if p.requires_grad),
                            sum(p.numel() for p in attr_clf.parameters() if p.requires_grad)]

    print("num traiable params:", num_trainable_params)
    
    loss_list = []
    forward_time_list = []
    asp_time_list = []
    gradient_time_list = []
    backward_time_list = []

    train_recall_list = []
    val_recall_list = []
    test_recall_list = []

    sm_per_batch_list = []

    best_test_recall = torch.zeros((1), dtype=torch.float)
    best_val_recall = torch.zeros((1), dtype=torch.float)


    #setup dataloader for the datasets
    all_oid = np.arange(0,NUM_OBJECTS)
    object_string = "".join([ f"object({oid1},{oid2}). " for oid1 in all_oid for oid2 in all_oid if oid1 != oid2])
    object_string = "".join(["".join([f"object({oid}). " for oid in all_oid]), object_string])

    #parse the SLASH program
    print("parse all train programs")
    program = "".join([KG, RULES, object_string, name_npp, relation_npp, attribute_npp, program_example])
    SLASHobj = SLASH(program, nnMapping, optimizers)

    for e in range(0, args.epochs):

        print("---TRAIN---")
        loss, forward_time, gradient_time, asp_time, backward_time, sm_per_batch = SLASHobj.learn(dataset_loader=train_loader,
            epoch=e, method="same", k_num=0, p_num=args.p_num, same_threshold={"relation":0.999, "name":0.999, "attr":0.99}, writer=writer, 
            vqa=True, batched_pass=True, vqa_params= vqa_params)

        loss_list.append(loss)
        forward_time_list.append(forward_time)
        asp_time_list.append(asp_time)
        gradient_time_list.append(gradient_time)
        backward_time_list.append(backward_time)
        sm_per_batch_list.append(sm_per_batch)

        time_metrics = {"forward_time":forward_time_list,
                        "asp_time":asp_time_list,
                        "backward_time":backward_time_list,
                        "gradient_time":gradient_time_list,
                        "sm_per_batch":sm_per_batch_list
                        }

        writer.add_scalar('train/loss', loss, e)

        saveModelPath = 'data/'+exp_name+'/slash_vqa_models_seed'+str(args.seed)+'_epoch_'+str(e)+'.pt'
        # Export results and networks
        print('Storing the trained model into {}'.format(saveModelPath))

        # print("---TRAIN---")
        # recall_5_train = SLASHobj.testVQA(train_loader_test_mode, args.p_num, vqa_params=vqa_params)
        # writer.add_scalar('train/recall', recall_5_train, e)
        # print("recall@5 train", recall_5_train)

        print("---VAL---")
        recall_5_val, val_time = SLASHobj.testVQA(val_loader, args.p_num, vqa_params=vqa_params)
        writer.add_scalar('val/recall', recall_5_val, e)
        val_recall_list.append(recall_5_val)
        print("val--recall@5:", recall_5_val, ", val_time:", val_time )
        val_recall_list.append(recall_5_val)


        print("---TEST---")
        if type(test_f) == str:
            recall_5_test, test_time = SLASHobj.testVQA(test_loader, args.p_num, vqa_params=vqa_params)
            writer.add_scalar('test/recall', recall_5_test, e)
            print("test-recall@5", recall_5_test)
            test_recall_list.append(recall_5_test)


        elif type(test_f) == dict:
            test_time = 0
            recalls = []
            for key in test_f:
                recall_5_test, tt = SLASHobj.testVQA(test_f[key], args.p_num, vqa_params=vqa_params)
                test_time += tt
                recalls.append(recall_5_test)
                print("test-recall@5_{}".format(key), recall_5_test, ", test_time:", tt )
                writer.add_scalar('test/recall_{}'.format(key), recall_5_test, e)
            writer.add_scalar('test/recall_c_all', np.array(recalls).mean(), e)
            print("test-recall@5_c_all", np.mean(recalls), ", test_time:", test_time)

            test_recall_list.append(recalls)
                


        # Check if the new best value for recall@5 on val dataset is reached.
        # If so, then save the new best performing model and test performance
        if best_val_recall < recall_5_val:
            best_val_recall = recall_5_val

            print("Found the new best performing model and store it!")
            saveBestModelPath = 'data/'+exp_name+'/slash_vqa_models_seed'+str(args.seed)+'best.pt'
            torch.save({"relation_clf": rela_clf.state_dict(),
                    "name_clf":  name_clf.state_dict(),
                    "attr_clf":attr_clf.state_dict(),
                    "resume": {
                        "optimizer_rela":optimizers['relation'].state_dict(),
                        "optimizer_name":optimizers['name'].state_dict(),
                        "optimizer_attr":optimizers['attr'].state_dict(),
                        "epoch":e
                            },
                    "args":args,
                    "loss": loss_list,
                    "train_recall_list":train_recall_list,
                    "val_recall_list":val_recall_list,
                    "test_recall_list":test_recall_list,
                    "time_metrics":time_metrics,
                    "program":program,
                    "vqa_params":vqa_params}, saveBestModelPath)


        print("storing the model")
        torch.save({"relation_clf": rela_clf.state_dict(),
                    "name_clf":  name_clf.state_dict(),
                    "attr_clf":attr_clf.state_dict(),
                    "resume": {
                        "optimizer_rela":optimizers['relation'].state_dict(),
                        "optimizer_name":optimizers['name'].state_dict(),
                        "optimizer_attr":optimizers['attr'].state_dict(),
                        "epoch":e
                            },
                    "args":args,
                    "loss": loss_list,
                    "train_recall_list":train_recall_list,
                    "val_recall_list":val_recall_list,
                    "test_recall_list":test_recall_list,
                    "time_metrics":time_metrics,
                    "program":program,
                    "vqa_params":vqa_params}, saveModelPath)
            
        # Update the RTPT
        rtpt.step(subtitle=f"loss={loss:2.2f};recall@5={recall_5_test:2.2f}")
        
if __name__ == "__main__":
    slash_vqa()
