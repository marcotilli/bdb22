# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:10:08 2021

@author: r10p86
"""

import torch
import torch.nn as nn
#import torch.nn.functional as F


#https://medium.com/dataseries/visualizing-the-feature-maps-and-filters-by-convolutional-neural-networks-e1462340518e
#https://pub.towardsai.net/tuning-pytorch-hyperparameters-with-optuna-470edcfd4dc
#https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f

#https://www.frontiersin.org/articles/10.3389/fmolb.2019.00044/full


def calc_outsize(H, W, kern, pad=(0,0), dil=(1,1), stride=(2,2)):
    #H_out = (H + 2 * pad[0] - dil[0] * (kern[0] - 1) - 1) / stride[0] + 1
    #W_out = (W + 2 * pad[1] - dil[1] * (kern[1] - 1) - 1) / stride[1] + 1
    H_out = (H - dil[0] * (kern[0] - 1) - 1) / stride[0] + 1
    W_out = (W - dil[1] * (kern[1] - 1) - 1) / stride[1] + 1
    H_out = np.round(H_out - 0.5, 0)
    W_out = np.round(W_out - 0.5, 0)
    return (H_out, W_out)

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
                        nn.Conv2d(n_chan, 16, (5,3)),
                        nn.MaxPool2d(3),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, (5,3)),
                        nn.MaxPool2d(3),
                        nn.ReLU(),
                        #nn.Conv2d(16, 32, 3),
                        #nn.ReLU(),
                        #nn.MaxPool2d(3)
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
        
        


