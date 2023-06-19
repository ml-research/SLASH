"""
The source code is based on:
NeurASP: Embracing Neural Networks into Answer Set Programming
Zhun Yang, Adam Ishay, Joohyung Lee. Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence
Main track. Pages 1755-1762.

https://github.com/azreasoners/NeurASP
"""


import re
import sys
import time

import clingo
import torch
import torch.nn as nn

import numpy as np
import time

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../SLASH/')

from operator import itemgetter
from mvpp import MVPP
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
from joblib import Parallel, delayed
import scipy 

from itertools import count

def pad_3d_tensor(target, framework, bs, ruleNum, max_classes):
    '''
    Turns a 3d list with unequal length into a padded tensor or 3D numpy array 
    
    @param target: nested list
    @param framework: type of the framework (either numpy or pytorch) of the output
    @param bs: len of the first dimension of the list
    @param ruleNum: max len of the second dimension of the list
    @param max_classes:  max len of the third dimension of the list
    :return:returns a tensor or 3D numpy array
    '''

    if framework == 'numpy':
        padded = torch.tensor([np.hstack((np.asarray(row, dtype=np.float32),  [0] * (max_classes - len(row))) ) for batch in target for row in batch]).type(torch.FloatTensor).view(bs, ruleNum, max_classes)    
    
    if framework == 'torch':
        padded = torch.stack([torch.hstack((row,  torch.tensor([0] * (max_classes - len(row)), device="cuda" ) ) ) for batch in target for row in batch ]).view(ruleNum, bs, max_classes)
    return padded


def replace_plus_minus_occurences(pi_prime):
    """
    For a given query/program replace all occurences of the +- Notation with the ASP readable counterpart eg. digit(0, +i1, -0) -> digit(0, 1, i1, 0)

    @param pi_prime: input query/program as string
    :return:returns the ASP readable counterpart of the given query/program and a dict of all used operators per npp
    """


    pat_pm = r'(\s*[a-z][a-zA-Z0-9_]*)\((\s*[A-Z]*[a-zA-Z0-9_]*\s*),\s*\+([A-Z]*[a-zA-Z0-9_]*\s*),\s*\-([A-Z]*[a-zA-Z0-9_]*\s*)\)' #1 +- p(c|x) posterior
    pat_mp = r'(\s*[a-z][a-zA-Z0-9_]*)\((\s*[A-Z]*[a-zA-Z0-9_]*\s*),\s*\-([A-Z]*[a-zA-Z0-9_]*\s*),\s*\+([A-Z]*[a-zA-Z0-9_]*\s*)\)' #2 -+ p(x|c) likelihood
    pat_mm = r'(\s*[a-z][a-zA-Z0-9_]*)\((\s*[A-Z]*[a-zA-Z0-9_]*\s*),\s*\-([A-Z]*[a-zA-Z0-9_]*\s*),\s*\-([A-Z]*[a-zA-Z0-9_]*\s*)\)' #3 -- p(x,c) joint
    pat_pp = r'(\s*[a-z][a-zA-Z0-9_]*)\((\s*[A-Z]*[a-zA-Z0-9_]*\s*),\s*\+([A-Z]*[a-zA-Z0-9_]*\s*),\s*\+([A-Z]*[a-zA-Z0-9_]*\s*)\)' #3 -- p(c) prior

    #pattern for vqa relations with two vars
    pat_pm2 = r'([a-z][a-zA-Z0-9_]*)\((\s*[A-Z]*[a-zA-Z0-9_]*\s*),\s*\+\(([a-zA-Z0-9_ ]*,[a-zA-Z0-9_ ]*)\),\s*\-([A-Z]*[a-zA-Z0-9_]*\s*)\)'

    #track which operator(+-,-+,--) was used for which npp
    npp_operators= {}
    for match in re.findall(pat_pm, pi_prime):
        if match[0] not in npp_operators:
            npp_operators[match[0]] = {}
        npp_operators[match[0]][1] = True
    for match in re.findall(pat_mp, pi_prime):
        if match[0] not in npp_operators:
            npp_operators[match[0]] = {}
        npp_operators[match[0]][2] = True
    for match in re.findall(pat_mm, pi_prime):
        if match[0] not in npp_operators:
            npp_operators[match[0]] = {}
        npp_operators[match[0]][3] = True
    for match in re.findall(pat_pp, pi_prime):
        if match[0] not in npp_operators:
            npp_operators[match[0]] = {}
        npp_operators[match[0]][4] = True

    for match in re.findall(pat_pm2, pi_prime):
        if match[0] not in npp_operators:
            npp_operators[match[0]] = {}
        npp_operators[match[0]][1] = True
    
    #replace found matches with asp compatible form for npp occuring in rules
    #example: digit(0,+A,-N1) -> digit(0,1,A,N1)
    pi_prime = re.sub( pat_pm, r'\1(\2,1,\3,\4)', pi_prime)
    pi_prime = re.sub( pat_mp, r'\1(\2,2,\3,\4)', pi_prime)
    pi_prime = re.sub( pat_mm, r'\1(\2,3,\3,\4)', pi_prime)
    pi_prime = re.sub( pat_pp, r'\1(\2,4,\3,\4)', pi_prime)

    pi_prime = re.sub( pat_pm2, r'\1(\2,1,\3,\4)', pi_prime)

    return pi_prime, npp_operators

def compute_models_splitwise(big_M_split, query_batch_split, dmvpp, method, k, same_threshold, obj_filter, vqa_params):
        """
        Computes the potential solutions and P(Q) for a split of a batch.
        
        @param big_M_split:
        @param query_batch_split:
        @param dmvpp:
        @param method:
        @param k: 
        @same_threshold:
        @obj_filter:
        @vqa_params: 
        :return: returns the potential solutions, the atom indices for the gradient computation and the probability P(Q)
        """

        query_batch_split = query_batch_split.tolist()
        

        
        #create a list to store the stable models into
        model_batch_list_split = []
        models_idx_list = []

        #iterate over all queries
        for bidx, query in enumerate(query_batch_split):
            
            #produce gradients for every data entry
            dmvpp.M = big_M_split[:,bidx,:]
            dmvpp.normalize_M()

            #if len(dmvpp.parameters[ruleIdx]) == 1:
                #    dmvpp.parameters[ruleIdx] =  [dmvpp.parameters[ruleIdx][0][0],1-dmvpp.parameters[ruleIdx][0][0]]

            query, _ = replace_plus_minus_occurences(query)
            

            #exact (default): uses all/k-first stable models for WMC 
            if method == 'exact': 
                models = dmvpp.find_all_SM_under_query(query)

            #uses only one most probable stable model for WMC  
            elif method == 'most_prob':
                models = dmvpp.find_k_most_probable_SM_under_query_noWC(query, k=1)

            #uses the k most probable stables models for WMC
            elif method == 'top_k':
                models = dmvpp.find_k_most_probable_SM_under_query_noWC(query,k=k)
                #gradients = dmvpp.mvppLearn(models)

            #reduces the npp grounding to the most probable instantiations
            elif method == "same":
                if obj_filter is not None: #in the vqa case we want to filter out objects which are not in the image
                    models = dmvpp.find_SM_vqa_with_same(query, k=k, threshold=same_threshold, obj_filter=obj_filter[bidx], vqa_params= vqa_params)
                else:
                    models = dmvpp.find_SM_with_same(query, k=k, threshold=same_threshold)
                

            #reduces the npp grounding to the most probable instantiations and then uses the top k stable models for WMC
            elif method == "same_top_k":
                models = dmvpp.find_k_most_probable_SM_with_same(query,k=k)
            else:
                print('Error: the method \'%s\' should be either \'exact\', \'most_prob\',\'top_k\',\'same_top_k\'  or \'same\'', method)
    
            #old NeurASP code - Can also be reenabled in SLASH
            # #samples atoms from the npp until k stable models are found
            # elif method == 'sampling':
            #     models = dmvpp.sample_query(query, num=k)

            # elif method == 'network_prediction':
            #     models = dmvpp.find_k_most_probable_SM_under_query_noWC(query, k=1)
            #     check = SLASH.satisfy(models[0], mvpp['program_asp'] + query)
            #     if check:
            #         continue
            # elif method == 'penalty':
            #     models = dmvpp.find_all_SM_under_query()
            #     models_noSM = [model for model in models if not SLASH.satisfy(model, mvpp['program_asp'] + query)]
                        
            models_idx = dmvpp.collect_atom_idx(models)

            models_idx_list.append(models_idx)
            model_batch_list_split.append(models)

        return model_batch_list_split, models_idx_list


def compute_vqa_splitwise(big_M_split, query_batch_split, dmvpp, obj_filter_split, k, pred_vector_size, vqa_params):
    """
    Computes the gradients, stable models and P(Q) for a split of a batch.
    
    @param networkOutput_split:
    @param query_batch_split:
    @param mvpp:
    @param n:
    @param normalProbs:
    @param dmvpp:
    @param method:
    @param opt:
    @param k: 
    :return:returns the gradients, the stable models and the probability P(Q)
    """

    query_batch_split = query_batch_split.tolist()
    
    
    #create a list to store the target predictions into
    pred_batch_list_split = []
    
    
    #iterate over all queries
    for bidx, query in enumerate(query_batch_split):
        
        dmvpp.M = big_M_split[:,bidx,:]
        dmvpp.normalize_M()

        #if len(dmvpp.parameters[ruleIdx]) == 1:
            #    dmvpp.parameters[ruleIdx] =  [dmvpp.parameters[ruleIdx][0][0],1-dmvpp.parameters[ruleIdx][0][0]]



        #query, _ = replace_plus_minus_occurences(query)

        #find all targets with same
        models = dmvpp.find_SM_vqa_with_same(query, k=k, obj_filter= obj_filter_split[bidx], threshold={"relation":1, "name":1, "attr":1}, train=False, vqa_params=vqa_params)

        #compute probabilites of targets in the model
        # first retrieve the object ids of targets in the model  
        pred_vec = np.zeros(pred_vector_size)

        #if we have a model
        if len(models) > 0:
            targets = {}
            models = [model for model in models if model != []] #remove empty models

            #go through all models and check for targets
            for model in models:    
                if len(model)> 1:   
                    model_prob = 0
                    model_has_npp_atom = False
                    for atom in model:
                        
                        #we found the target atom and save the id of the object
                        if re.match(r'target', atom):
                            #add target id
                            target_id = int(re.match(r'target\(([0-9]*)\)',atom)[1])
                        else:
                            #if the non target atom is a ground npp atom get its probability and add it 
                            if atom in dmvpp.ga_map:
                                rule_idx, atom_idx = dmvpp.ga_map[atom]
                                model_prob += dmvpp.M[rule_idx, atom_idx].log()
                                model_has_npp_atom = True

                    # only add model prob for real probability values < 1
                    if model_has_npp_atom:
                        #if this is the first model with a target of that id
                        if target_id not in targets:
                            targets[target_id] = []
                        targets[target_id].append(model_prob)

            #get the maximum value for each target object
            for target_id in targets:
                pred_vec[target_id] = torch.tensor(targets[target_id]).max().exp()
        
        #add prediction vector entry for this query to the batch list containing all queries
        pred_batch_list_split.append(pred_vec)
    return pred_batch_list_split



class SLASH(object):
    def __init__(self, dprogram, networkMapping, optimizers, gpu=True):

        """
        @param dprogram: a string for a NeurASP program
        @param networkMapping: a dictionary maps network names to neural networks modules
        @param optimizers: a dictionary maps network names to their optimizers
        
        @param gpu: a Boolean denoting whether the user wants to use GPU for training and testing
        """

        # the neural models should be run on a GPU for better performance. The gradient computation is vectorized but in some cases it makes sense
        # to run them on the cpu. For example in the case of the MNIST addition we create a lot of small tensors and putting them all on the gpu is significant overhead
        # while the computation itself is comparable for small tensors. 
        # As a rule of thumb you can say that with more date or more npps it makes more sense to use the gpu for fast gradient computation.
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
        self.grad_comp_device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

        self.dprogram = dprogram
        self.const = {} # the mapping from c to v for rule #const c=v.
        self.n = {} # the mapping from network name to an integer n denoting the domain size; n would be 1 or N (>=3); note that n=2 in theory is implemented as n=1
        self.max_n = 2 # integer denoting biggest domain of all npps
        self.e = {} # the mapping from network name to an integer e
        self.domain = {} # the mapping from network name to the domain of the predicate in that network atom
        self.normalProbs = None # record the probabilities from normal prob rules
        self.networkOutputs = {}
        self.networkGradients = {}
        self.networkTypes = {}
        if gpu==True:
            self.networkMapping = {key : nn.DataParallel(networkMapping[key].to(self.device)) for key in networkMapping}
        else:
            self.networkMapping = networkMapping
        self.optimizers = optimizers
        # self.mvpp is a dictionary consisting of 4 keys: 
        # 1. 'program': a string denoting an MVPP program where the probabilistic rules generated from network are followed by other rules;
        # 2. 'networkProb': a list of lists of tuples, each tuple is of the form (model, i ,term, j)
        # 3. 'atom': a list of list of atoms, where each list of atoms is corresponding to a prob. rule
        # 4. 'networkPrRuleNum': an integer denoting the number of probabilistic rules generated from network
        self.mvpp = {'networkProb': [], 'atom': [], 'networkPrRuleNum': 0,'networkPrRuleNumWithoutBinary': 0, 'binary_rule_belongings':{}, 'program': '','networkProbSinglePred':{}}
        self.mvpp['program'], self.mvpp['program_pr'], self.mvpp['program_asp'], self.npp_operators = self.parse(query='')
        self.stableModels = [] # a list of stable models, where each stable model is a list
        self.prob_q = [] # a list of probabilites for each query in the batch


    def constReplacement(self, t):
        """ Return a string obtained from t by replacing all c with v if '#const c=v.' is present

        @param t: a string, which is a term representing an input to a neural network
        """
        t = t.split(',')
        t = [self.const[i.strip()] if i.strip() in self.const else i.strip() for i in t]
        return ','.join(t)

    def networkAtom2MVPPrules(self, networkAtom, npp_operators):
        """
        @param networkAtom: a string of a neural atom
        @param countIdx: a Boolean value denoting whether we count the index for the value of m(t, i)[j]
        """

        # STEP 1: obtain all information
        regex = '^(npp)\((.+)\((.+)\),\((.+)\)\)$'
        out = re.search(regex, networkAtom)        
        
        network_type = out.group(1)
        m = out.group(2)
        e, inf_type, t = out.group(3).split(',', 2) # in case t is of the form t1,...,tk, we only split on the second comma
        domain = out.group(4).split(',')
        inf_type =int(inf_type)

        #TODO how do we get the network type?
        self.networkTypes[m] = network_type

        t = self.constReplacement(t)
        # check the value of e
        e = e.strip()
        e = int(self.constReplacement(e))
        n = len(domain)
        if n == 2:
            n = 1
        self.n[m] = n
        if self.max_n <= n:
            self.max_n = n
        self.e[m] = e

        self.domain[m] = domain
        if m not in self.networkOutputs:
            self.networkOutputs[m] = {}

        for o in npp_operators[m]:
            if o not in self.networkOutputs[m]:
                self.networkOutputs[m][o] = {}
            self.networkOutputs[m][o][t]= None    
        
        # STEP 2: generate MVPP rules
        mvppRules = []

        # we have different translations when n = 2 (i.e., n = 1 in implementation) or when n > 2
        if n == 1:
            for i in range(e):
                rule = '@0.0 {}({}, {}, {}, {}); @0.0 {}({},{}, {}, {}).'.format(m, i, inf_type, t, domain[0], m, i, inf_type, t, domain[1])
                prob = [tuple((m, i, inf_type, t, 0))]
                atoms = ['{}({}, {}, {}, {})'.format(m, i,inf_type, t, domain[0]), '{}({},{},{}, {})'.format(m, i, inf_type,  t, domain[1])]
                mvppRules.append(rule)
                self.mvpp['networkProb'].append(prob)
                self.mvpp['atom'].append(atoms)
                self.mvpp['networkPrRuleNum'] += 1
                self.mvpp['networkPrRuleNumWithoutBinary'] += 1

            self.mvpp['networkProbSinglePred'][m] = True
        elif n > 2:
            if m == "attr": #TODO special case hack to map a model to multiple npp's as in the attributes in vqa
                self.mvpp['binary_rule_belongings'][self.mvpp['networkPrRuleNum']] = (m,t)
            for i in range(e):
                rule = ''
                prob = []
                atoms = []
                for j in range(n):
                    atom = '{}({},{}, {}, {})'.format(m,  i, inf_type, t, domain[j])
                    rule += '@0.0 {}({},{}, {}, {}); '.format(m, i,inf_type, t, domain[j])
                    prob.append(tuple((m, i, inf_type, t, j)))
                    atoms.append(atom)

            mvppRules.append(rule[:-2]+'.')
            self.mvpp['networkProb'].append(prob)
            self.mvpp['atom'].append(atoms)
            self.mvpp['networkPrRuleNum'] += 1
            self.mvpp['networkPrRuleNumWithoutBinary'] += 1

            
        else:
            print('Error: the number of element in the domain %s is less than 2' % domain)
        return mvppRules



    def parse(self, query=''):
        dprogram = self.dprogram + query
        # 1. Obtain all const definitions c for each rule #const c=v.
        regex = '#const\s+(.+)=(.+).'
        out = re.search(regex, dprogram)
        if out:
            self.const[out.group(1).strip()] = out.group(2).strip()
            
        # 2. Generate prob. rules for grounded network atoms
        clingo_control = clingo.Control(["--warn=none"])
        
        # 2.1 remove weak constraints and comments
        program = re.sub(r'\n:~ .+\.[ \t]*\[.+\]', '\n', dprogram)
        program = re.sub(r'\n%[^\n]*', '\n', program)
        
        # 2.2 replace [] with ()
        program = program.replace('[', '(').replace(']', ')')
        

        # 2.3 use MVPP package to parse prob. rules and obtain ASP counter-part
        mvpp = MVPP(program)
        #if mvpp.parameters and not self.normalProbs:
        #    self.normalProbs = mvpp.parameters
        pi_prime = mvpp.pi_prime



        #2.4 parse +-Notation and add a const to flag the operation in the npp call
        pi_prime = pi_prime.replace(' ','').replace('#const','#const ')
        
        #replace all occurences of the +- calls
        pi_prime, npp_operators = replace_plus_minus_occurences(pi_prime)


        #extend npps definitions with the operators found
        #example: npp(digit(1,X),(0,1,2,3,4,5,6,7,8,9)):-img(X). with a +- and -- call in the program 
        # -> npp(digit(1,3,X),(0,1,2,3,4,5,6,7,8,9)):-img(X). npp(digit(1,1,X),(0,1,2,3,4,5,6,7,8,9)):-img(X).
        #pat_npp = r'(npp\()([a-z]*[a-zA-Z0-9_]*)(\([0-9]*,)([A-Z]*[a-zA-Z0-9_]*\),\((?:[a-z0-9_]*,)*[a-z0-9_]*\)\))(:-[a-z][a-zA-Z0-9_]*\([A-Z][a-zA-Z0-9_]*\))?.'

        #can match fact with arity > 1
        pat_npp = r'(npp\()([a-z]*[a-zA-Z0-9_]*)(\([0-9]*,)([A-Z]*)([A-Z]*[a-zA-Z0-9_]*\),\((?:[a-z0-9_]*,)*[a-z0-9_]*\)\))(:-[a-z][a-zA-Z0-9_]*\([A-Z][a-zA-Z0-9_,]*\))?.'


        def npp_sub(match):
            """
            filters program for npp occurances
            """
            npp_extended =""
            for o in npp_operators[match.group(2)]:

                
                if match.group(6) is None:
                    body = ""
                    var_replacement = match.group(4)
                
                elif re.match(r':-[a-z]*\([0-9a-zA-Z_]*,[0-9a-zA-Z_]*\)', match.group(6)):
                    body = match.group(6)
                    var_replacement = re.sub(r'(:-)([a-zA-Z0-9]*)\(([a-z0-9A-Z]*,[a-z0-9A-Z]*)\)', r"\3" ,body)
                else: 
                    body = match.group(6)
                    var_replacement = match.group(4)

                npp_extended = '{}{}{}{}{}{}{}{}.\n'.format(match.group(1), match.group(2),match.group(3),o,",", var_replacement,match.group(5), body)+ npp_extended 
            return npp_extended

        pi_prime = re.sub(pat_npp, npp_sub, pi_prime)

        # 2.5 use clingo to generate all grounded network atoms and turn them into prob. rules
        clingo_control.add("base", [], pi_prime)
        clingo_control.ground([("base", [])])
        #symbolic mappings map the constants to functions in the ASP program
        symbols = [atom.symbol for atom in clingo_control.symbolic_atoms]
        

        #iterate over all NPP atoms and extract information for the MVPP program
        mvppRules = [self.networkAtom2MVPPrules(str(atom),npp_operators) for atom in symbols if (atom.name == 'npp')]
        mvppRules = [rule for rules in mvppRules for rule in rules]
        
        # 3. obtain the ASP part in the original NeurASP program after +- replacements
        lines = [line.strip() for line in pi_prime.split('\n') if line and not re.match("^\s*npp\(", line)]

        return '\n'.join(mvppRules + lines), '\n'.join(mvppRules), '\n'.join(lines), npp_operators


    @staticmethod
    def satisfy(model, asp):
        """
        Return True if model satisfies the asp program; False otherwise
        @param model: a stable model in the form of a list of atoms, where each atom is a string
        @param asp: an ASP program (constraints) in the form of a string
        """
        asp_with_facts = asp + '\n'
        for atom in model:
            asp_with_facts += atom + '.\n'
        clingo_control = clingo.Control(['--warn=none'])
        clingo_control.add('base', [], asp_with_facts)
        clingo_control.ground([('base', [])])
        result = clingo_control.solve()
        if str(result) == 'SAT':
            return True
        return False

        
    def infer(self, dataDic, query='', mvpp='',  dataIdx=None):
        """
        @param dataDic: a dictionary that maps terms to tensors/np-arrays
        @param query: a string which is a set of constraints denoting a query
        @param mvpp: an MVPP program used in inference
        """

        mvppRules = ''
        facts = ''

        # Step 1: get the output of each neural network
        for m in self.networkOutputs:
            self.networkMapping[m].eval()
            
            for o in self.networkOutputs[m]:
                for t in self.networkOutputs[m][o]:

                    if dataIdx is not None:
                        dataTensor = dataDic[t][dataIdx]
                    else:
                        dataTensor = dataDic[t]
                    
                    self.networkOutputs[m][o][t] = self.networkMapping[m](dataTensor).view(-1).tolist()

        for ruleIdx in range(self.mvpp['networkPrRuleNum']):
            
            probs = [self.networkOutputs[m][inf_type][t][i*self.n[m]+j] for (m, i, inf_type, t, j) in self.mvpp['networkProb'][ruleIdx]]
            #probs = [self.networkOutputs[m][inf_type][t][0][i*self.n[m]+j] for (m, i, inf_type, t, j) in self.mvpp['networkProb'][ruleIdx]]

            if len(probs) == 1:
                mvppRules += '@{:.15f} {}; @{:.15f} {}.\n'.format(probs[0], self.mvpp['atom'][ruleIdx][0], 1 - probs[0], self.mvpp['atom'][ruleIdx][1])
            else:
                tmp = ''
                for atomIdx, prob in enumerate(probs):
                    tmp += '@{:.15f} {}; '.format(prob, self.mvpp['atom'][ruleIdx][atomIdx])
                mvppRules += tmp[:-2] + '.\n'
        
        # Step 3: find an optimal SM under query
        dmvpp = MVPP(facts + mvppRules + mvpp)
        return dmvpp.find_k_most_probable_SM_under_query_noWC(query, k=1)


    def split_network_outputs(self, big_M, obj_filter, query_batch, p_num):
        if len(query_batch)< p_num:
            if type(query_batch) == tuple:
                p_num = len(query_batch)
            else:
                p_num=query_batch.shape[0]

        #partition dictionary for different processors                
        
        splits = np.arange(0, int(p_num))
        partition = int(len(query_batch) / p_num)
        partition_mod = len(query_batch) % p_num 
        partition = [partition]*p_num
        partition[-1]+= partition_mod
        
        query_batch_split = np.split(query_batch,np.cumsum(partition))[:-1]

        if obj_filter is not  None:
            obj_filter = np.array(obj_filter)
            obj_filter_split = np.split(obj_filter, np.cumsum(partition))[:-1]
        else:
            obj_filter_split = None 

        big_M_splits = torch.split(big_M, dim=1, split_size_or_sections=partition)


        return big_M_splits, query_batch_split, obj_filter_split, p_num


 
    def learn(self, dataset_loader, epoch, method='exact', opt=False, k_num=0, p_num=1, slot_net=None, hungarian_matching=False, vqa=False, marginalisation_masks=None,  writer=None, same_threshold=0.99, batched_pass=False, vqa_params=None):
        
        """
        @param dataset_loader: a pytorch dataloader object returning a dictionary e.g {im1: [bs,28,28], im2:[bs,28,28]} and the queries
        @param epoch: an integer denoting the current epoch
        @param method: a string in {'exact', 'same',''} denoting whether the gradients are computed exactly or by sampling
        @param opt: stands for optimal -> if true we select optimal stable models
        @param k_num: select k stable models, default k_num=0 means all stable models
        @param p_num: a positive integer denoting the number of processor cores to be used during the training
        @param slot_net: a slot attention network for the set prediction experiment mapping from an image to slots
        @param vqa: VQA option for the vqa experiment. Enables VQA specific datahandling
        @param hungarian_matching: boolean denoting wherever the matching is done in the LP or if the results of the hungarian matching should be integrated in the LP
        @param marginalisation_masks: a list entailing one marginalisation mask for each batch of dataList
        @param writer: Tensorboard writer for plotting metrics
        @param same_threshold: Threshold when method = same or same_top_k. Can be either a scalar value or a dict mapping treshold to networks m {"digit":0.99}
        @param batched_pass: boolean to forward all t belonging to the same m in one pass instead of t passes
        """
        
        assert p_num >= 1 and isinstance(p_num, int), 'Error: the number of processors used should greater equals one and a natural number'

        # get the mvpp program by self.mvpp
        #old NeurASP code. Can be reanbled if needed
        #if method == 'network_prediction':
        #    dmvpp = MVPP(self.mvpp['program_pr'], max_n= self.max_n)
        #elif method == 'penalty':
        #    dmvpp = MVPP(self.mvpp['program_pr'], max_n= self.max_n)

        #add the npps to the program now or later
        if method == 'same' or method == 'same_top_k':
            dmvpp = MVPP(self.mvpp['program'], prob_ground=True , binary_rule_belongings= self.mvpp['binary_rule_belongings'], max_n= self.max_n)
        else:
            dmvpp = MVPP(self.mvpp['program'], max_n= self.max_n)

    
        # we train all neural network models
        for m in self.networkMapping:
            self.networkMapping[m].train() #torch training mode
            self.networkMapping[m].module.train() #torch training mode

    
        total_loss = []
        sm_per_batch_list = []
        #store time per epoch
        forward_time = []
        asp_time = []
        gradient_time  = []
        backward_time = []
        
        #iterate over all batches
        pbar = tqdm(total=len(dataset_loader))
        for batch_idx, (batch) in enumerate(dataset_loader):      
            start_time = time.time()

            if hungarian_matching:
                data_batch, query_batch, obj_enc_batch = batch
            elif vqa:
                data_batch, query_batch, obj_filter, _ = batch
            else:
                data_batch, query_batch = batch


                    
            # If we have marginalisation masks, than we have to pick one for the batch
            if marginalisation_masks is not None:
                marg_mask = marginalisation_masks[i]
            else:
                marg_mask = None
            
            #STEP 0: APPLY SLOT ATTENTION TO TRANSFORM IMAGE TO SLOTS
            #we have a map which is : im: im_data
            #we want a map which is : s1: slot1_data, s2: slot2_data, s3: slot3_data
            if slot_net is not None:
                slot_net.train()
                dataTensor_after_slot = slot_net(data_batch['im'].to(self.device)) #forward the image

                #add the slot outputs to the data batch
                for slot_num in range(slot_net.n_slots):
                    key = 's'+str(slot_num+1)
                    data_batch[key] = dataTensor_after_slot[:,slot_num,:]
                        

                
            #data is a dictionary. we need to edit its key if the key contains a defined const c
            #where c is defined in rule #const c=v.
            data_batch_keys = list(data_batch.keys())              
            for key in data_batch_keys:
                data_batch[self.constReplacement(key)] = data_batch.pop(key)
                        
            
            # Step 1: get the output of each network and initialize the gradients
            networkOutput = {}
            networkLLOutput = {}

            #iterate over all networks            
            for m in self.networkOutputs:
                if m not in networkOutput:
                    networkOutput[m] = {}


                #iterate over all output types and forwarded the input t trough the network
                networkLLOutput[m] = {}
                for o in self.networkOutputs[m]: 
                    if o not in networkOutput[m]:
                        networkOutput[m][o] = {}
                                        
                    
                    #one forward pass to get the outputs
                    if self.networkTypes[m] == 'npp':
                        
                        #forward all t belonging to the same m in one pass instead of t passes
                        if batched_pass:

                            #collect all inputs t for every network m 
                            dataTensor = [data_batch.get(key).to(device=self.device) for key in self.networkOutputs[m][o].keys()]
                            dataTensor = torch.cat(dataTensor)

                            len_keys = len(query_batch)


                            output = self.networkMapping[m].forward(
                                                            dataTensor.to(self.device),
                                                            marg_idx=marg_mask,
                                                            type=o)

                            outputs = output.split(len_keys)

                            networkOutput[m][o] = {**networkOutput[m][o], **dict(zip(self.networkOutputs[m][o].keys(), outputs))}
                        
                        else:
                            for t in self.networkOutputs[m][o]:

                                dataTensor = data_batch[t]                                    
                                
                                #we have a list of data elements but want a Tensor of the form [batchsize,...]
                                if isinstance(dataTensor, list):
                                    dataTensor = torch.stack(dataTensor).squeeze(dim=1) #TODO re-enable

                                #foward the data
                                networkOutput[m][o][t] = self.networkMapping[m].forward(
                                                                                dataTensor.to(self.device),
                                                                                marg_idx=marg_mask,
                                                                                type=o)

                                #if the network predicts only one probability we add a placeholder for the false class
                                if m in self.mvpp['networkProbSinglePred']:
                                    networkOutput[m][o][t] = torch.stack((networkOutput[m][o][t], torch.zeros_like(networkOutput[m][o][t])), dim=-1)



                    #store the outputs of the neural networks as a class variable
                    self.networkOutputs[m][o] = networkOutput[m][o] #this is of shape [first batch entry, second batch entry,...]

            
            #match the outputs with the hungarian matching and create a predicate to add to the logic program
            #NOTE: this is a hacky solution which leaves open the question wherever we can have neural mappings in our logic program
            if hungarian_matching is True:
                
                obj_batch = {}
                if "shade" in networkOutput:
                    obj_batch['color'] = obj_enc_batch[:,:,0:9] # [500, 4,20] #[c,c,c,c,c,c,c,c,c , s,s,s,s , h,h,h, z,z,z, confidence]
                    obj_batch['shape'] = obj_enc_batch[:,:,9:13]
                    obj_batch['shade'] = obj_enc_batch[:,:,13:16]
                    obj_batch['size'] = obj_enc_batch[:,:,16:19]

                    concepts = ['color', 'shape', 'shade','size']
                    slots = ['s1', 's2', 's3','s4']
                    num_obs = 4
                
                else:        
                    obj_batch['size'] = obj_enc_batch[:,:,0:3] # [500, 4,20] #[c,c,c,c,c,c,c,c,c , s,s,s,s , h,h,h, z,z,z, confidence]
                    obj_batch['material'] = obj_enc_batch[:,:,3:6]
                    obj_batch['shape'] = obj_enc_batch[:,:,6:10]
                    obj_batch['color'] = obj_enc_batch[:,:,10:19]

                    concepts = ['color', 'shape', 'material','size']
                    slots = ['s1', 's2', 's3','s4','s5','s6','s7','s8','s9','s10']
                    num_obs = 10
                    
                kl_cost_matrix = torch.zeros((num_obs,num_obs,len(query_batch)))

                #build KL cost matrix
                for obj_id in range(0,num_obs):
                    for slot_idx, slot in enumerate(slots):
                        summed_kl = 0
                        for concept in concepts:
                            b = obj_batch[concept][:,obj_id].type(torch.FloatTensor)
                            a = networkOutput[concept][1][slot].detach().cpu()
                            summed_kl += torch.cdist(a[:,None,:],b[:,None,:]).squeeze()

                        kl_cost_matrix[obj_id, slot_idx] = summed_kl 

                kl_cost_matrix = np.einsum("abc->cab", kl_cost_matrix.cpu().numpy())


                indices = np.array(
                list(map(scipy.optimize.linear_sum_assignment, kl_cost_matrix)))
                

                def slot_name_comb(x):
                    return ''.join(["slot_name_comb(o{}, s{}). ".format(i[0]+1, i[1]+1)  for i in x])

                assignments = np.array(list(map(slot_name_comb, np.einsum("abc->acb",indices))))
                query_batch = list(map(str.__add__, query_batch, assignments))

            
            #stack all nn outputs in matrix M
            big_M = torch.zeros([ len(self.mvpp['networkProb']),len(query_batch), self.max_n], device=self.grad_comp_device)

            c = 0
            for m in networkOutput:
                for o in networkOutput[m]:
                    for t in networkOutput[m][o]:
                        big_M[c,:, :networkOutput[m][o][t].shape[1]]= networkOutput[m][o][t].detach().to('cpu')
                        c+=1

            
            #set all matrices in the dmvpp class to the cpu for model computation
            dmvpp.put_selection_mask_on_device('cpu')
            big_M = big_M.to(device='cpu')
            dmvpp.M =  dmvpp.M.to(device='cpu')

            #normalize the SLASH copy of M 
            #big_M = normalize_M(big_M, dmvpp.non_binary_idx, dmvpp.selection_mask)


            step1 = time.time()
            forward_time.append(step1 - start_time)
                        
            #### Step 2: compute stable models and the gradients

            #we split the network outputs such that we can put them on different processes
            if vqa: 
                big_M_splits, query_batch_split, obj_filter_split, p_num = self.split_network_outputs(big_M, obj_filter, query_batch, p_num)
            else:
                big_M_splits, query_batch_split, _, p_num = self.split_network_outputs(big_M, None, query_batch, p_num)
                obj_filter_split = [None] * len(query_batch_split)



            split_outputs = Parallel(n_jobs=p_num,backend='loky')( #backend='loky')(
                delayed(compute_models_splitwise)
                (
                    big_M_splits[i].detach(), query_batch_split[i],
                        dmvpp, method, k_num, same_threshold, obj_filter_split[i], vqa_params
                )
                        for i in range(p_num))

            del big_M_splits

            #concatenate potential solutions, the atom indices and query p(Q) from all splits back into a single batch
            model_batch_list_splits = []
            models_idx_list = []

            #collect the models and model computation times
            for i in range(p_num):
                model_batch_list_splits.extend(split_outputs[i][0])
                models_idx_list.extend(split_outputs[i][1])   #batch, model, atoms

            #amount of Stable models used for gradient computations per batch
            sm_per_batch = np.sum([ len(sm_batch)  for splits in model_batch_list_splits for sm_batch in splits ])
            sm_per_batch_list.append(sm_per_batch)

            #save stable models
            try:
                model_batch_list = np.concatenate(model_batch_list_splits)

                self.stableModels = model_batch_list

            except ValueError as e:
                pass
                #print("fix later")
                #print(e)
                #for i in range(0, len(model_batch_list_splits)):
                #    print("NUM:",i)
                #    print(model_batch_list_splits[i])
            

            step2 = time.time()
            asp_time.append(step2 - step1)

            #compute gradients
            dmvpp.put_selection_mask_on_device(self.grad_comp_device)
            big_M = big_M.to(device=self.grad_comp_device)

            gradients_batch_list = []
            for bidx in count(start=0, step=1):
                if bidx < model_batch_list_splits.__len__():
                    dmvpp.M = big_M[:,bidx,:]
                    dmvpp.normalize_M()
                    gradients_batch_list.append(dmvpp.mvppLearn(model_batch_list_splits[bidx], models_idx_list[bidx], self.grad_comp_device))
                else:
                    break
                
            del big_M

            #stack all gradients
            gradient_tensor = torch.stack(gradients_batch_list)

            #store the gradients, the stable models and p(Q) of the last batch processed
            self.networkGradients = gradient_tensor


            # Step 3: update parameters in neural networks
            step3 = time.time()
            gradient_time.append(step3 - step2)

            
            networkOutput_stacked = torch.zeros([gradient_tensor.shape[1], gradient_tensor.shape[0], gradient_tensor.shape[2]], device=torch.device('cuda:0'))
            gradient_tensor = gradient_tensor.swapaxes(0,1)
            org_idx = (gradient_tensor.sum(dim=2) != 0)#.flatten(0,1)

            #add all NN outputs which have a gradient into tensor for backward pass
            c = 0
            for m in networkOutput:
                for o in networkOutput[m]:
                    for t in networkOutput[m][o]:
                        for bidx in range(0,org_idx.shape[1]): #iterate over batch
                            if org_idx[c, bidx]:
                                networkOutput_stacked[c,bidx, :networkOutput[m][o][t].shape[1]]= networkOutput[m][o][t][bidx]
                        c+=1


            #multiply every probability with its gradient
            gradient_tensor.requires_grad=True
            result = torch.einsum("abc, abc -> abc", gradient_tensor.to(device='cuda'),networkOutput_stacked)

            not_used_npps = org_idx.sum() / (result.shape[0]* result.shape[1])
            result = result[result.abs() != 0 ].sum()

            #in the case vqa case we need to only use the npps over existing objects
            if vqa:
                total_obj = 0 
                for of in obj_filter:
                    total_obj += 2* of + of * (of-1)
                not_used_npps = org_idx.sum() / total_obj

            
            #get the number of discrete properties, e.g. sum all npps times the atoms entailing the npps 
            sum_discrete_properties = sum([len(listElem) for listElem in self.mvpp['atom']]) * len(query_batch)
            sum_discrete_properties = torch.Tensor([sum_discrete_properties]).to(device=self.device)

            #scale to actualy used npps
            sum_discrete_properties = sum_discrete_properties * not_used_npps

            #get the mean over the sum of discrete properties
            result_ll = result / sum_discrete_properties

            #backward pass
            #for gradient descent we minimize the negative log likelihood    
            result_nll = -result_ll
            
            #reset optimizers
            for  m in self.optimizers:
                self.optimizers[m].zero_grad()


            #append the loss value
            total_loss.append(result_nll.cpu().detach().numpy())
            
            #backward pass
            result_nll.backward(retain_graph=True)


            
            #apply gradients with each optimizer
            for m in self.optimizers:
                self.optimizers[m].step()


            last_step = time.time()
            backward_time.append(last_step - step3)

                    
            if writer is not None:
                writer.add_scalar('train/loss_per_batch', result_nll.cpu().detach().numpy(), batch_idx+epoch*len(dataset_loader))
                writer.add_scalar('train/sm_per_batch', sm_per_batch, batch_idx+epoch*len(dataset_loader))

            pbar.update()
        pbar.close()
        
        if writer is not None:

            writer.add_scalar('train/forward_time', np.sum(forward_time), epoch)
            writer.add_scalar('train/asp_time', np.sum(asp_time), epoch)
            writer.add_scalar('train/gradient_time', np.sum(gradient_time), epoch)
            writer.add_scalar('train/backward_time', np.sum(backward_time), epoch)
            writer.add_scalar('train/sm_per_epoch', np.sum(sm_per_batch_list), epoch)


        print("avg loss over batches:", np.mean(total_loss))
        print("1. forward time: ", np.sum(forward_time))
        print("2. asp time:", np.sum(asp_time))
        print("3. gradient time:", np.sum(gradient_time))
        print("4. backward time: ", np.sum(backward_time))
        print("SM processed", np.sum(sm_per_batch_list))



        return np.mean(total_loss), forward_time, asp_time, gradient_time, backward_time, sm_per_batch_list


            

    def testNetwork(self, network, testLoader, ret_confusion=False):
        """
        Return a real number in [0,100] denoting accuracy
        @network is the name of the neural network or probabilisitc circuit to check the accuracy. 
        @testLoader is the input and output pairs.
        """
        self.networkMapping[network].eval()
        # check if total prediction is correct
        correct = 0
        total = 0
        # check if each single prediction is correct
        singleCorrect = 0
        singleTotal = 0
        
        #list to collect targets and predictions for confusion matrix
        y_target = []
        y_pred = []
        with torch.no_grad():
            for data, target in testLoader:
                                
                output = self.networkMapping[network](data.to(self.device))
                if self.n[network] > 2 :
                    pred = output.argmax(dim=-1, keepdim=True) # get the index of the max log-probability
                    target = target.to(self.device).view_as(pred)
                    
                    correctionMatrix = (target.int() == pred.int()).view(target.shape[0], -1)
                    y_target = np.concatenate( (y_target, target.int().flatten().cpu() ))
                    y_pred = np.concatenate( (y_pred , pred.int().flatten().cpu()) )
                    
                    
                    correct += correctionMatrix.all(1).sum().item()
                    total += target.shape[0]
                    singleCorrect += correctionMatrix.sum().item()
                    singleTotal += target.numel()
                else: 
                    pred = np.array([int(i[0]<0.5) for i in output.tolist()])
                    target = target.numpy()
                    
                    
                    correct += (pred.reshape(target.shape) == target).sum()
                    total += len(pred)
        accuracy = correct / total

        if self.n[network] > 2:
            singleAccuracy = singleCorrect / singleTotal
        else:
            singleAccuracy = 0
        
        if ret_confusion:
            confusionMatrix = confusion_matrix(np.array(y_target), np.array(y_pred))
            return accuracy, singleAccuracy, confusionMatrix

        return accuracy, singleAccuracy
    
    # We interprete the most probable stable model(s) as the prediction of the inference mode
    # and check the accuracy of the inference mode by checking whether the query is satisfied by the prediction
    def testInferenceResults(self, dataset_loader):
        """ Return a real number in [0,1] denoting the accuracy
        @param dataset_loader: a dataloader object loading a dataset to test on
        """

        correct = 0
        len_dataset = 0
        #iterate over batch
        for data_batch, query_batch in dataset_loader:
            len_dataset += len(query_batch)

            #iterate over each entry in batch
            for dataIdx in range(0, len(query_batch)):
                models = self.infer(data_batch, query=':- mistake.', mvpp=self.mvpp['program_asp'],  dataIdx= dataIdx)

                query,_ =  replace_plus_minus_occurences(query_batch[dataIdx])

                for model in models:
                    if self.satisfy(model, query):
                        correct += 1
                        break

        accuracy = 100. * correct / len_dataset
        return accuracy


    def testConstraint(self, dataset_loader, mvppList):
        """
        @param dataList: a list of dictionaries, where each dictionary maps terms to tensors/np-arrays
        @param queryList: a list of strings, where each string is a set of constraints denoting a query
        @param mvppList: a list of MVPP programs (each is a string)
        """

        # we evaluate all nerual networks
        for func in self.networkMapping:
            self.networkMapping[func].eval()

        # we count the correct prediction for each mvpp program
        count = [0]*len(mvppList)


        len_data = 0
        for data_batch, query_batch in dataset_loader:
            len_data += len(query_batch)

            # data is a dictionary. we need to edit its key if the key contains a defined const c
            # where c is defined in rule #const c=v.
            data_batch_keys = list(data_batch.keys())
            for key in data_batch_keys:
                data_batch[self.constReplacement(key)] = data_batch.pop(key)

            # Step 1: get the output of each neural network

            for m in self.networkOutputs:
                for o in self.networkOutputs[m]: #iterate over all output types and forwarded the input t trough the network
                    for t in self.networkOutputs[m][o]:
                        self.networkOutputs[m][o][t] = self.networkMapping[m].forward(
                                                                                    data_batch[t].to(self.device),
                                                                                    marg_idx=None,
                                                                                    type=o)

            # Step 2: turn the network outputs into a set of ASP facts            
            aspFactsList = []
            for bidx in range(len(query_batch)):
                aspFacts = ''
                for ruleIdx in range(self.mvpp['networkPrRuleNum']):
                    
                    #get the network outputs for the current element in the batch and put it into the correct rule
                    probs = [self.networkOutputs[m][inf_type][t][bidx][i*self.n[m]+j] for (m, i, inf_type, t, j) in self.mvpp['networkProb'][ruleIdx]]

                    if len(probs) == 1:
                        atomIdx = int(probs[0] < 0.5) # t is of index 0 and f is of index 1
                    else:
                        atomIdx = probs.index(max(probs))
                    aspFacts += self.mvpp['atom'][ruleIdx][atomIdx] + '.\n'
                aspFactsList.append(aspFacts)
        

            # Step 3: check whether each MVPP program is satisfied
            for bidx in range(len(query_batch)):
                for programIdx, program in enumerate(mvppList):

                    query,_ =  replace_plus_minus_occurences(query_batch[bidx])
                    program,_ = replace_plus_minus_occurences(program)

                    # if the program has weak constraints
                    if re.search(r':~.+\.[ \t]*\[.+\]', program) or re.search(r':~.+\.[ \t]*\[.+\]', query):
                        choiceRules = ''
                        for ruleIdx in range(self.mvpp['networkPrRuleNum']):
                            choiceRules += '1{' + '; '.join(self.mvpp['atom'][ruleIdx]) + '}1.\n'


                        mvpp = MVPP(program+choiceRules)
                        models = mvpp.find_all_opt_SM_under_query_WC(query=query)
                        models = [set(model) for model in models] # each model is a set of atoms
                        targetAtoms = aspFacts[bidx].split('.\n')
                        targetAtoms = set([atom.strip().replace(' ','') for atom in targetAtoms if atom.strip()])
                        if any(targetAtoms.issubset(model) for model in models):
                            count[programIdx] += 1
                    else:
                        mvpp = MVPP(aspFacts[bidx] + program)
                        if mvpp.find_one_SM_under_query(query=query):
                            count[programIdx] += 1
        for programIdx, program in enumerate(mvppList):
            print('The accuracy for constraint {} is {}({}/{})'.format(programIdx+1, float(count[programIdx])/len_data,float(count[programIdx]),len_data ))

            
            
    
    def forward_slot_attention_pipeline(self, slot_net, dataset_loader):
        """
        Makes one forward pass trough the slot attention pipeline to obtain the probabilities/log likelihoods for all classes for each object. 
        The pipeline includes  the SlotAttention module followed by probabilisitc circuits for probabilites for the discrete properties.
        @param slot_net: The SlotAttention module
        @param dataset_loader: Dataloader containing a shapeworld/clevr dataset to be forwarded
        """
        with torch.no_grad():

            probabilities = {}  # map to store all output probabilities(posterior)
            slot_map = {} #map to store all slot module outputs
            
            for data_batch, _,_ in dataset_loader:
                
                #forward img to get slots
                dataTensor_after_slot = slot_net(data_batch['im'].to(self.device))#SHAPE [BS,SLOTS, SLOTSIZE]
                
                #dataTensor_after_slot has shape [bs, num_slots, slot_vector_length]
                _, num_slots ,_ = dataTensor_after_slot.shape 

                
                for sdx in range(0, num_slots):
                    slot_map["s"+str(sdx)] = dataTensor_after_slot[:,sdx,:]
                

                #iterate over all slots and forward them through all nets (shape + color + ... )
                for key in slot_map:
                    if key not in probabilities:
                        probabilities[key] = {}
                    
                    for network in self.networkMapping:
                        posterior= self.networkMapping[network].forward(slot_map[key])#[BS, num_discrete_props]
                        if network not in probabilities[key]:
                            probabilities[key][network] = posterior
                        else: 
                            probabilities[key][network] = torch.cat((probabilities[key][network], posterior))


            return probabilities

    def get_recall(self, logits, labels,  topk=3):


        # Calculate the recall
        _, pred_idx = logits.topk(topk, 1, True, True) #get idx of the biggest k values in the prediction tensor


        #gather gets the elements from a given indices tensor. Here we get the elements at the same positions from our (top5) predictions
        #we then sum all entries up along the axis and therefore count the top-5 entries in the labels tensors at the prediction position indices
        correct = torch.sum(labels.gather(1, pred_idx), dim=1)

        #now we sum up all true labels. We clamp them to be maximum top5
        correct_label = torch.clamp(torch.sum(labels, dim = 1), 0, topk)

        #we now can compare if the number of correctly found top-5 labels on the predictions vector is the same as on the same positions as the GT vector
        #if the num of gt labels is zero (question has no answer) then we get a nan value for the div by 0 -> we replace this with recall 1 
        recall = torch.mean(torch.nan_to_num(correct / correct_label,1)).item()

        return recall



    def testVQA(self, test_loader, p_num = 1, k=5, vqa_params= None):
        """
        @param test_loader: a dataloader object 
        @param p_num: integer denoting the number of processes to split the batches on
        @param k: denotes how many targets pred are used for the recall metric
        """

        start_test = time.time()
        dmvpp = MVPP(self.mvpp['program'],prob_ground=True,binary_rule_belongings= self.mvpp['binary_rule_belongings'])
        networkOutput = {}

        # we train all neural network models
        for m in self.networkMapping:
            self.networkMapping[m].eval() #torch training mode
            self.networkMapping[m].module.eval() #torch training mode

        pred_vector_list = []
        gt_vector_list = []

        #iterate over all batches
        #pbar = tqdm(total=len(test_loader))
        for batch_idx, (batch) in enumerate(test_loader):            
    
            data_batch, query_batch, obj_filter, target = batch

            # Step 1: get the output of each neural network
            for m in self.networkOutputs:
                if m not in networkOutput:
                    networkOutput[m] = {}

                #iterate over all inference types o and forwarded the input t trough the network
                for o in self.networkOutputs[m]: 
                    if o not in networkOutput[m]:
                        networkOutput[m][o] = {}
                                        
                    if self.networkTypes[m] == 'npp':

                        #collect all inputs t for every network m 
                        dataTensor = [data_batch.get(key).to(device=self.device) for key in self.networkOutputs[m][o].keys()]
                        dataTensor = torch.cat(dataTensor)
                        len_keys = len(query_batch)

                        output = self.networkMapping[m].forward(
                                                        dataTensor.to(self.device),
                                                        marg_idx=None,
                                                        type=o)

                        outputs = output.split(len_keys)
                        networkOutput[m][o] = {**networkOutput[m][o], **dict(zip(self.networkOutputs[m][o].keys(), outputs))}



            #stack all nn outputs in matrix M
            big_M = torch.zeros([ len(self.mvpp['networkProb']),len(query_batch), self.max_n], device=self.grad_comp_device)


            c = 0
            for m in networkOutput:
                for o in networkOutput[m]:
                    for t in networkOutput[m][o]:
                        big_M[c,:, :networkOutput[m][o][t].shape[1]]= networkOutput[m][o][t].detach().to(self.grad_comp_device)
                        c+=1


            big_M_splits, query_batch_split, obj_filter_split, p_num = self.split_network_outputs(big_M, obj_filter, query_batch, p_num)


            # Step 2: compute stable models using the probabilities
            # we get a list of pred vectors where each index represents if the object is a target and its corresponding target probability 

            dmvpp.put_selection_mask_on_device('cpu')
            pred_vector_list_split = Parallel(n_jobs=p_num,backend='loky')( #backend='loky')(
                delayed(compute_vqa_splitwise)
                (
                    big_M_splits[i].detach().cpu(), query_batch_split[i],
                        dmvpp,  obj_filter_split[i], k, target.shape[1], vqa_params
                )
                        for i in range(p_num))
            
            #stack together the target vectors from the different procceses
            pred_vector_list.append(torch.stack([torch.tensor(i) for p in pred_vector_list_split for i in p ]))
            gt_vector_list.append(target.clone().detach())
            #pbar.update()
        #pbar.close()


        #stack together the target vectors from the different batches
        pred_vector = torch.cat(pred_vector_list, dim=0)
        gt_vector = torch.cat(gt_vector_list, dim=0)



        #compute the recall
        recall = self.get_recall(pred_vector, gt_vector, 5)
        #print("RECALL",recall)

        return recall, time.time() - start_test
