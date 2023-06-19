#!/usr/bin/env python
# coding: utf-8


import train
import datetime


#Python script to start the shapeworld4 slot attention experiment
#Define your experiment(s) parameters as a hashmap having the following parameters
example_structure = {'experiment_name': 
                   {'structure': 'poon-domingos',
                    'pd_num_pieces': [4],
                    'lr': 0.01, #the learning rate to train the SPNs with, the slot attention module has a fixed lr=0.0004  
                    'bs':50, #the batchsize
                    'epochs':1000, #number of epochs to train
                    'lr_warmup':True, #boolean indicating the use of learning rate warm up
                    'lr_warmup_steps':25, #number of epochs to warm up the slot attention module, warmup does not apply to the SPNs
                    'start_date':"01-01-0001", #current date
                    'resume':False, #you can stop the experiment and set this parameter to true to load the last state and continue learning
                    'credentials':'DO', #your credentials for the rtpt class
                    'hungarian_matching': True,
                    'explanation': """Running the whole SlotAttention+Slash pipeline using poon-domingos as SPN structure learner."""}}




#EXPERIMENTS
date_string = datetime.datetime.today().strftime('%d-%m-%Y')


for seed in [0,1,2,3,4]:
    experiments = {'shapeworld4': 
                   {'structure': 'poon-domingos', 'pd_num_pieces': [4],
                    'lr': 0.01, 'bs':512, 'epochs':1000, 
                    'lr_warmup_steps':8, 'lr_decay_steps':360,
                    'start_date':date_string, 'resume':False, 'credentials':'DO','seed':seed,
                    'p_num':16, 'method':'same_top_k', 'hungarian_matching': False,
                    'explanation': """Running the whole SlotAttention+Slash pipeline using poon-domingos as SPN structure learner."""}
                    }
    

    print("shapeworld4")
    train.slash_slot_attention("shapeworld4", experiments["shapeworld4"])





