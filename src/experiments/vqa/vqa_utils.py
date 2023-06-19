"""
The source code is based on:
Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning
Jiani Huang, Ziyang Li, Binghong Chen, Karan Samel, Mayur Naik, Le Song, Xujie Si
Advances in Neural Information Processing Systems 34 (NeurIPS 2021)
https://proceedings.neurips.cc/paper/2021/hash/d367eef13f90793bd8121e2f675f0dc2-Abstract.html
"""

import os
import json
import argparse
import numpy as np
import torch
import random
import pickle
from sklearn import metrics

TIME_OUT = 120

###########################################################################
# basic utilities


def to_image_id_dict(dict_list):
    ii_dict = {}
    for dc in dict_list:
        image_id = dc['image_id']
        ii_dict[image_id] = dc
    return ii_dict


articles = ['a', 'an', 'the', 'some', 'it']


def remove_article(st):
    st = st.split()
    st = [word for word in st if word not in articles]
    return " ".join(st)

################################################################################################################
# AUC calculation

def get_recall(labels, logits, topk=2):
    #a = torch.tensor([[0, 1, 1], [1, 1, 0], [0, 0, 1],[1, 1, 1]])
    #b = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1],[0, 1, 1]])

    # Calculate the recall
    _, pred = logits.topk(topk, 1, True, True) #get idx of the biggest k values in the tensor
    print(pred)

    pred = pred.t() #transpose

    #gather gets the elements from a given indices tensor. Here we get the elements at the same positions from our (top5) predictions
    #we then sum all entries up along the axis and therefore count the top-5 entries in the labels tensors at the prediction position indices
    correct = torch.sum(labels.gather(1, pred.t()), dim=1)
    print("correct", correct)

    #now we some up all true labels. We clamp them to be maximum top5
    correct_label = torch.clamp(torch.sum(labels, dim = 1), 0, topk)

    #we now can compare if the number of correctly  found top-5 labels on the predictions vector is the same as on the same positions as the GT vector
    print("correct label", correct_label)

    accuracy = torch.mean(correct / correct_label).item()

    return accuracy

def single_auc_score(label, pred):
    label = label
    pred = [p.item() if not (type(p) == float or type(p) == int)
            else p for p in pred]
    if len(set(label)) == 1:
        auc = -1
    else:
        fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    return auc


def auc_score(labels, preds):
    if type(labels) == torch.Tensor:
        labels = labels.long().cpu().detach().numpy()
    if type(preds) == torch.Tensor:
        preds = preds.cpu().detach().numpy()

    if (type(labels) == torch.Tensor or type(labels) == np.ndarray) and len(labels.shape) == 2:
        aucs = []
        for label, pred in zip(labels, preds):
            auc_single = single_auc_score(label, pred)
            if not auc_single < 0:
                aucs.append(auc_single)
        auc = np.array(aucs).mean()
    else:
        auc = single_auc_score(labels, preds)

    return auc


def to_binary_labels(labels, attr_num):
    if labels.shape[-1] == attr_num:
        return labels

    binary_labels = torch.zeros((labels.shape[0], attr_num))
    for ct, label in enumerate(labels):
        binary_labels[ct][label] = 1
    return binary_labels


##############################################################################
# model loading
def get_default_args():

    DATA_ROOT = os.path.abspath(os.path.join(
        os.path.abspath(__file__), "../../data"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--feat_dim', type=int, default=2048)
    # parser.add_argument('--type', default='name')
    parser.add_argument('--model_dir', default=DATA_ROOT + '/model_ckpts')
    parser.add_argument('--meta_f', default=DATA_ROOT + '/preprocessing/gqa_info.json')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    return args


def gen_task(formatted_scene_graph_path, vg_image_data_path, questions_path, features_path, n=100, q_length=None):

    with open(questions_path, 'rb') as questions_file:
        questions = pickle.load(questions_file)

    with open(formatted_scene_graph_path, 'rb') as vg_scene_graphs_file:
        scene_graphs = pickle.load(vg_scene_graphs_file)

    with open(vg_image_data_path, 'r') as vg_image_data_file:
        image_data = json.load(vg_image_data_file)

    features = np.load(features_path, allow_pickle=True)
    features = features.item()

    image_data = to_image_id_dict(image_data)

    for question in questions[:n]:

        if q_length is not None:
            current_len = len(question['clauses'])
            if not current_len == q_length:
                continue

        # functions = question["clauses"]
        image_id = question["image_id"]
        scene_graph = scene_graphs[image_id]
        cur_image_data = image_data[image_id]

        if not scene_graph['image_id'] == image_id or not cur_image_data['image_id'] == image_id:
            raise Exception("Mismatched image id")

        info = {}
        info['image_id'] = image_id
        info['scene_graph'] = scene_graph
        info['url'] = cur_image_data['url']
        info['question'] = question
        info['object_feature'] = []
        info['object_ids'] = []

        for obj_id in scene_graph['names'].keys():
            info['object_ids'].append(obj_id)
            info['object_feature'].append(features.get(obj_id))

        yield(info)

