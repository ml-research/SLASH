"""
The source code is based on:
Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning
Jiani Huang, Ziyang Li, Binghong Chen, Karan Samel, Mayur Naik, Le Song, Xujie Si
Advances in Neural Information Processing Systems 34 (NeurIPS 2021)
https://proceedings.neurips.cc/paper/2021/hash/d367eef13f90793bd8121e2f675f0dc2-Abstract.html
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
torch.autograd.set_detect_anomaly(True)   

sys.path.append('../../EinsumNetworks/src/')
from einsum_wrapper import EiNet


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, n_layers, dropout_rate, softmax):
        super(MLPClassifier, self).__init__()

        self.softmax = softmax
        self.output_dim = output_dim

        layers = []
        layers.append(nn.Linear(input_dim, latent_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(latent_dim))
        #layers.append(nn.InstanceNorm1d(latent_dim))
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(latent_dim))
            #layers.append(nn.InstanceNorm1d(latent_dim))
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(latent_dim, output_dim))


        self.net = nn.Sequential(*layers)

    def forward(self, x, marg_idx=None, type=None):
        if x.sum() == 0:
            return torch.ones([x.shape[0], self.output_dim], device='cuda')

        idx = x.sum(dim=1)!=0 # get idx of true objects
        logits = torch.zeros(x.shape[0], self.output_dim, device='cuda')

        logits[idx] = self.net(x[idx])
        

        if self.softmax:
            probs = F.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)

        return probs

# FasterCNN object feature size
feature_dim = 2048 

name_clf = MLPClassifier(
    input_dim=feature_dim,
    output_dim=500,
    latent_dim=1024,
    n_layers=2,
    dropout_rate=0.3,
    softmax=True
)

rela_clf = MLPClassifier(
    input_dim=(feature_dim+4)*2,  
    output_dim=229,       
    latent_dim=1024,
    n_layers=1,
    dropout_rate=0.5,
    softmax=True
)


attr_clf = MLPClassifier(
    input_dim=feature_dim,
    output_dim=609,
    latent_dim=1024,
    n_layers=1,
    dropout_rate=0.3,
    softmax=False
)



name_einet = EiNet(structure = 'poon-domingos',
                      pd_num_pieces = [4],
                      use_em = False,
                      num_var = 2048,
                      class_count = 500,
                      pd_width = 32,
                      pd_height = 64,
                      learn_prior = True)
rela_einet = EiNet(structure = 'poon-domingos',
                      pd_num_pieces = [4],
                      use_em = False,
                      num_var = 4104,
                      class_count = 229,
                      pd_width = 72,
                      pd_height = 57,
                      learn_prior = True)

attr_einet = EiNet(structure = 'poon-domingos',
                      pd_num_pieces = [4],
                      use_em = False,
                      num_var = 2048,
                      class_count = 609,
                      pd_width = 32,
                      pd_height = 64,
                      learn_prior = True)