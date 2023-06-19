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




import utils
from utils import set_manual_seed
from pathlib import Path
from rtpt import RTPT
import pickle


from knowledge_graph import RULES, KG
from dataGen import name_npp, relation_npp, attribute_npp
from models import name_clf, rela_clf, attr_clf 

print("...done")




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks"
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
    rtpt = RTPT(name_initials=args.credentials, experiment_name='SLASH VQA', max_iterations=1)

    # Start the RTPT tracking
    rtpt.start()
    #writer = SummaryWriter(os.path.join("runs","vqa", str(args.seed)), purge_step=0)

    #exp_name = 'vqa3'
    #Path("data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)
    #saveModelPath = 'data/'+exp_name+'/slash_vqa_models_seed'+str(args.seed)+'.pt'


    #TODO workaround that adds +- notation 
    program_example = """
%scallop conversion rules
name(O,N) :-  name(0,+O,-N).
attr(O,A) :-  attr(0, +O, -A).
relation(O1,O2,N) :-  relation(0, +(O1,O2), -N).
"""

    #test_f = "dataset/task_list/test_tasks_c3_1000.pkl"  # Test datset

    test_f = {"c2":"dataset/task_list/test_tasks_c2_1000.pkl",
              "c3":"dataset/task_list/test_tasks_c3_1000.pkl",
              "c4":"dataset/task_list/test_tasks_c4_1000.pkl",
              "c5":"dataset/task_list/test_tasks_c5_1000.pkl",
              "c6":"dataset/task_list/test_tasks_c6_1000.pkl"
              }
    

    num_obj = []
    if type(test_f) == str: 
        num_obj.append(determine_max_objects(test_f))
    #if we have multiple test files
    elif type(test_f) == dict:
        for key in test_f:   
            num_obj.append(determine_max_objects(test_f[key]))


    NUM_OBJECTS = np.max(num_obj)
    NUM_OBJECTS = 70


    vqa_params = {"l":200,
            "l_split":100,
            "num_names":500,
            "max_models":10000,
            "asp_timeout": 60}



    #load models #data/vqa18_10/slash_vqa_models_seed0_epoch_0.pt
    #src/experiments/vqa/
    #saved_models = torch.load("data/test/slash_vqa_models_seed42_epoch_9.pt")
    saved_models = torch.load("data/vqa_debug_relations_17_04_2023/slash_vqa_models_seed0_epoch_2.pt")

    print(saved_models.keys())
    rela_clf.load_state_dict(saved_models['relation_clf'])
    name_clf.load_state_dict(saved_models['name_clf'])
    attr_clf.load_state_dict(saved_models['attr_clf'])

    #create the SLASH Program , ,
    nnMapping = {'relation': rela_clf, 'name':name_clf , "attr":attr_clf}
    optimizers = {'relation': torch.optim.Adam(rela_clf.parameters(), lr=0.001, eps=1e-7),
                      'name': torch.optim.Adam(name_clf.parameters(), lr=0.001, eps=1e-7),
                      'attr': torch.optim.Adam(attr_clf.parameters(), lr=0.001, eps=1e-7)}



    all_oid = np.arange(0,NUM_OBJECTS)
    object_string = "".join([ f"object({oid1},{oid2}). " for oid1 in all_oid for oid2 in all_oid if oid1 != oid2])
    object_string = "".join(["".join([f"object({oid}). " for oid in all_oid]), object_string])

    #parse the SLASH program
    print("create SLASH program")
    program = "".join([KG, RULES, object_string, name_npp, relation_npp, attribute_npp, program_example])
    SLASHobj = SLASH(program, nnMapping, optimizers)

    #load the data
    if type(test_f) == str: 
        test_data = VQA("test", test_f, NUM_OBJECTS)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    #if we have multiple test files
    elif type(test_f) == dict: 
        for key in test_f:
            test_data = VQA("test", test_f[key], NUM_OBJECTS)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
            test_f[key] = test_loader



    print("---TEST---")
    if type(test_f) == str:
        recall_5_test, test_time = SLASHobj.testVQA(test_loader, args.p_num, vqa_params=vqa_params)
        print("test-recall@5", recall_5_test)

    elif type(test_f) == dict:
        test_time = 0
        recalls = []
        for key in test_f:
            recall_5_test, tt = SLASHobj.testVQA(test_f[key], args.p_num, vqa_params=vqa_params)
            test_time += tt
            recalls.append(recall_5_test)
            print("test-recall@5_{}".format(key), recall_5_test, ", test_time:", tt )
        print("test-recall@5_c_all", np.mean(recalls), ", test_time:", test_time)


if __name__ == "__main__":
    slash_vqa()
