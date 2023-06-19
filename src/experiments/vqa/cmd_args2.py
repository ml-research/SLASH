"""
The source code is based on:
Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning
Jiani Huang, Ziyang Li, Binghong Chen, Karan Samel, Mayur Naik, Le Song, Xujie Si
Advances in Neural Information Processing Systems 34 (NeurIPS 2021)
https://proceedings.neurips.cc/paper/2021/hash/d367eef13f90793bd8121e2f675f0dc2-Abstract.html
"""


import argparse
import sys
import random
import numpy as np
import torch
import os
import logging

# Utility function for take in yes/no and convert it to boolean
def convert_str_to_bool(cmd_args):
    for key, val in vars(cmd_args).items():
        if val == "yes":
            setattr(cmd_args, key, True)
        elif val == "no":
            setattr(cmd_args, key, False)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class LearningSetting(object):

    def __init__(self):
        data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
        dataset_dir = os.path.join(data_dir, "dataset")
        knowledge_base_dir = os.path.join(data_dir, "knowledge_base")

        self.parser = argparse.ArgumentParser(description='Argparser', allow_abbrev=True)
        # Logistics
        self.parser.add_argument("--seed", default=1234, type=int, help="set random seed")
        self.parser.add_argument("--gpu", default=0, type=int, help="GPU id")
        self.parser.add_argument('--timeout', default=600, type=int, help="Execution timeout")

        # Learning settings
        self.parser.add_argument("--name_threshold", default=0, type=float)
        self.parser.add_argument("--attr_threshold", default=0, type=float)
        self.parser.add_argument("--rela_threshold", default=0, type=float)
        # self.parser.add_argument("--name_topk_n", default=-1, type=int)
        # self.parser.add_argument("--attr_topk_n", default=-1, type=int)
        # self.parser.add_argument("--rela_topk_n", default=80, type=int)
        self.parser.add_argument("--topk", default=3, type=int)

        # training settings
        self.parser.add_argument('--feat_dim', type=int, default=2048)
        self.parser.add_argument('--n_epochs', type=int, default=20)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--max_workers', type=int, default=1)
        self.parser.add_argument('--axiom_update_size', type=int, default=4)
        self.parser.add_argument('--name_lr', type=float, default=0.0001)
        self.parser.add_argument('--attr_lr', type=float, default=0.0001)
        self.parser.add_argument('--rela_lr', type=float, default=0.0001)
        self.parser.add_argument('--reinforce', type=str2bool, nargs='?', const=True, default=False) # reinforce only support single thread
        self.parser.add_argument('--replays', type=int, default=5)

        self.parser.add_argument('--model_dir', default=data_dir+'/model_ckpts_sg')
        # self.parser.add_argument('--model_dir', default=None)
        self.parser.add_argument('--log_name', default='model.log')
        self.parser.add_argument('--feat_f', default=data_dir+'/features.npy')
        self.parser.add_argument('--train_f', default=dataset_dir+'/task_list/train_tasks_c2_10.pkl')
        self.parser.add_argument('--val_f', default=dataset_dir+'/task_list/val_tasks.pkl')
        self.parser.add_argument('--test_f', default=dataset_dir+'/task_list/test_tasks_c2_1000.pkl')
        self.parser.add_argument('--cul_prov', type=bool, default=False)

        self.parser.add_argument('--meta_f', default=data_dir+'/gqa_info.json')
        self.parser.add_argument('--scene_f', default=data_dir+'/gqa_formatted_scene_graph.pkl')
        self.parser.add_argument('--image_data_f', default=data_dir+'/image_data.json')
        self.parser.add_argument('--dataset_type', default='name') # name, relation, attr:<groupindex>

        self.parser.add_argument('--function', default=None) #KG_Find / Hypernym_Find / Find_Name / Find_Attr / Relate / Relate_Reverse / And / Or
        self.parser.add_argument('--knowledge_base_dir', default=knowledge_base_dir)
        # self.parser.add_argument('--interp_size', type=int, default=2)

        self.parser.add_argument('--save_dir', default=data_dir+"/problog_data")
        self.args = self.parser.parse_args(sys.argv[1:])

ls = LearningSetting()
cmd_args = ls.args
# print(cmd_args)

# Fix random seed for debugging purpose
if (ls.args.seed != None):
    random.seed(ls.args.seed)
    np.random.seed(ls.args.seed)
    torch.manual_seed(ls.args.seed)

if not type(cmd_args.gpu) == None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cmd_args.gpu)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

if cmd_args.model_dir is not None:
    if not os.path.exists(cmd_args.model_dir):
        os.makedirs(cmd_args.model_dir)

    log_path = os.path.join(cmd_args.model_dir, cmd_args.log_name)
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    logging.info(cmd_args)
    logging.info("start!")
