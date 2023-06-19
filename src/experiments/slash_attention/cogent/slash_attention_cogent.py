#!/usr/bin/env python
# coding: utf-8


import train
import numpy as np
import torch
import torchvision
import datetime

date_string = datetime.datetime.today().strftime('%d-%m-%Y')

#Python script to start the shapeworld4 slot attention experiment
#Define your experiment(s) parameters as a hashmap having the following parameters
example_structure = {'experiment_name': 
                   {'structure': 'poon-domingos',
                    'pd_num_pieces': [4],
                    'lr': 0.01, #the learning rate to train the SPNs with, the slot attention module has a fixed lr=0.0004  
                    'bs':512, #the batchsize
                    'epochs':1000, #number of epochs to train
                    'lr_warmup_steps':25, #number of epochs to warm up the slot attention module, warmup does not apply to the SPNs
                    'lr_decay_steps':100, #number of epochs it takes to decay to 50% of the specified lr
                    'start_date':"01-01-0001", #current date
                    'resume':False, #you can stop the experiment and set this parameter to true to load the last state and continue learning
                    'credentials':'AS', #your credentials for the rtpt class
                    'explanation': """Training on Condtion A, Testing on Condtion A and B to evaluate generalization of the model."""}}



experiments ={'shapeworld4_cogent_hung': 
                   {'structure': 'poon-domingos', 'pd_num_pieces': [4],
                    'lr': 0.01, 'bs':512, 'epochs':1000,
                    'lr_warmup_steps':8, 'lr_decay_steps':360,
                    'start_date':date_string, 'resume':False,
                    'credentials':'DO', 'seed':3, 'learn_prior':True,
                    'p_num':16, 'hungarian_matching':True, 'method':'probabilistic_grounding_top_k',
                    'explanation': """Training on Condtion A, Testing on Condtion A and B to evaluate generalization of the model."""}}



#train the network
for exp_name in experiments:
    print(exp_name)
    train.slash_slot_attention(exp_name, experiments[exp_name])





