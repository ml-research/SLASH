import torch
import torch.nn as nn   
import numpy as np
from EinsumNetwork import EinsumNetwork, Graph

device = torch.device('cuda:0')

#wrapper class to create an Einsum Network given a specific structure and parameters
class EiNet(EinsumNetwork.EinsumNetwork):
    def __init__(self , 
                 use_em,
                 structure = 'poon-domingos',
                 pd_num_pieces = [4],
                 depth = 8,
                 num_repetitions = 20,
                 num_var = 784,
                 class_count = 3,
                 K = 10,
                 num_sums = 10,
                 pd_height = 28,
                 pd_width = 28,
                 learn_prior = True
                ):
        

        # Structure
        self.structure = structure        
        self.class_count = class_count        
        classes = np.arange(class_count)  # [0,1,2,..,n-1]
        
        # Define the prior, i.e. P(C) and make it learnable.
        self.learnable_prior = learn_prior
        # P(C) is needed to apply the Bayes' theorem and to retrive
        # P(C|X) = P(X|C)*(P(C) / P(X)
        if self.class_count == 4:
            self.prior = torch.tensor([(1/3)*(2/3), (1/3)*(2/3), (1/3)*(2/3), (1/3)], dtype=torch.float, requires_grad=True, device=device).log()
        else:
            self.prior = torch.ones(class_count, device=device, dtype=torch.float)
        self.prior.fill_(1 / class_count)
        self.prior.log_()
        if self.learnable_prior:
            print("P(C) is learnable.")
            self.prior.requires_grad_()
        
        self.K = K
        self.num_sums = num_sums
        
        # 'poon-domingos'
        self.pd_num_pieces = pd_num_pieces  # [10, 28],[4],[7]
        self.pd_height = pd_height
        self.pd_width = pd_width

        
        # 'binary-trees'
        self.depth = depth
        self.num_repetitions = num_repetitions
        self.num_var = num_var
        
        # drop-out rate
        # self.drop_out = drop_out
        # print("The drop-out rate:", self.drop_out)
        
        # EM-settings
        self.use_em = use_em
        online_em_frequency = 1
        online_em_stepsize = 0.05  # 0.05
        print("train SPN with EM:",self.use_em)

        
        # exponential_family = EinsumNetwork.BinomialArray
        # exponential_family = EinsumNetwork.CategoricalArray
        exponential_family = EinsumNetwork.NormalArray
        
        exponential_family_args = None
        if exponential_family == EinsumNetwork.BinomialArray:
            exponential_family_args = {'N': 255}
        if exponential_family == EinsumNetwork.CategoricalArray:
            exponential_family_args = {'K': 1366120}
        if exponential_family == EinsumNetwork.NormalArray:
            exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}

        # Make EinsumNetwork
        if self.structure == 'poon-domingos':
            pd_delta = [[self.pd_height / d, self.pd_width / d] for d in self.pd_num_pieces]
            graph = Graph.poon_domingos_structure(shape=(self.pd_height, self.pd_width), delta=pd_delta)
        elif self.structure == 'binary-trees':
            graph = Graph.random_binary_trees(num_var=self.num_var, depth=self.depth, num_repetitions=self.num_repetitions)
        else:
            raise AssertionError("Unknown Structure")

    
        args = EinsumNetwork.Args(
                num_var=self.num_var,
                num_dims=1,
                num_classes=self.class_count,
                num_sums=self.num_sums,
                num_input_distributions=self.K,
                exponential_family=exponential_family,
                exponential_family_args=exponential_family_args,
                use_em=self.use_em,
                online_em_frequency=online_em_frequency,
                online_em_stepsize=online_em_stepsize)
        
        super().__init__(graph, args)
        super().initialize()
    
    def get_log_likelihoods(self, x):
        log_likelihood = super().forward(x)
        return log_likelihood

    def forward(self, x, marg_idx=None, type=1):
        
        # PRIOR
        if type == 4:
            expanded_prior = self.prior.expand(x.shape[0], self.prior.shape[0])
            return expanded_prior

        else:
            # Obtain P(X|C) in log domain
            if marg_idx:  # If marginalisation mask is passed
                self.set_marginalization_idx(marg_idx)
                log_likelihood = super().forward(x)
                self.set_marginalization_idx(None)
                likelihood = torch.nn.functional.softmax(log_likelihood, dim=1)
            else:
                log_likelihood = super().forward(x)

            #LIKELIHOOD
            if type == 2:
                likelihood = torch.nn.functional.softmax(log_likelihood, dim=1)
                # Sanity check for NaN-values
                if torch.isnan(log_likelihood).sum() > 0:
                    print("likelihood nan")
                
                return likelihood
            else:
                # Apply Bayes' Theorem to obtain P(C|X) instead of P(X|C)
                # as it is provided by the EiNet
                # 1. Computation of the prior, i.e. P(C), is already being
                # dealt with at the initialisation of the EiNet.
                # 2. Compute the normalization constant P(X)
                z = torch.logsumexp(log_likelihood + self.prior, dim=1)
                # 3. Compute the posterior, i.e. P(C|X) = (P(X|C) * P(C)) / P(X)
                posterior_log = (log_likelihood + self.prior - z[:, None])  # log domain
                #posterior = posterior_log.exp()  # decimal domain
            

            #POSTERIOR
            if type == 1:
                posterior = torch.nn.functional.softmax(posterior_log, dim=1)
                
                # Sanity check for NaN-values
                if torch.isnan(z).sum() > 0:
                    print("z nan")
                if torch.isnan(posterior).sum() > 0:
                    print("posterior nan")
                return posterior
            
            #JOINT
            elif type == 3:
                #compute the joint P(X|C) * P(C) 
                joint = torch.nn.functional.softmax(log_likelihood + self.prior, dim=1)

                return joint
