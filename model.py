# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:10:08 2021

@author: r10p86
"""

import torch
import torch.nn as nn
#import torch.nn.functional as F


# Hyperparameters
class hypers():
    def __init__(self):
        self.batch_size = 16
        self.n_epochs   = 1000
        self.layers_dict = {'n_conv_lays': 3, 
                   'c1':  8, 'c2': 16, 'c3': 32,
                   'ks1': 3, 'ks2': 3, 'ks3': 5
                   }    
        self.opimizer = torch.nn.optimizer.adam...
            
    def get_hyperparams(self):
        return self.n_epochs, self.batch_size, ... 
        
    def get_model_params(self):
        return self.layers_dict

    def optimizer(self):
           
        return opti

    def criterion_function(self):
        
        return crit_fct
        
# CNN Model
class CNN1_Team(nn.Module):
    def __init__(self, n_chan = 2, layers_dict = None):
        #self.layers_dict = layers_dict
        self.conv = nn.Sequential(
                        nn.Conv2d(n_chan, 8, 3),
                        nn.ReLU(),
                        nn.AvgPool2d(3),
                        nn.Conv2d(8, 16, 3),
                        nn.ReLU(),
                        nn.AvgPool2d(3),
                        nn.Conv2d(16, 32, 5),
                        nn.ReLU(),
                        nn.AvgPool2d(5)
                        )
        self.linear = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(32, 8),
                        nn.ReLU(),
                        nn.Linear(8, 1)         
                        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return(x)
        
        


