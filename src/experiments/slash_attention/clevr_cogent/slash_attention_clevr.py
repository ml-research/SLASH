#!/usr/bin/env python
# coding: utf-8



import train

import datetime





seed = 4
obj_num = 10
date_string = datetime.datetime.today().strftime('%d-%m-%Y')
experiments = {f'CLEVR{obj_num}_seed_{seed}': {'structure':'poon-domingos', 'pd_num_pieces':[4], 'learn_prior':True,
                         'lr': 0.01, 'bs':512, 'epochs':1000,
                        'lr_warmup_steps':8, 'lr_decay_steps':360, 'use_em':False, 'resume':False,
                        'method':'most_prob',
                         'start_date':date_string, 'credentials':'DO', 'p_num':16, 'seed':seed, 'obj_num':obj_num
                        }}


for exp_name in experiments:
    print(exp_name)
    train.slash_slot_attention(exp_name, experiments[exp_name])



