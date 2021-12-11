# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:10:08 2021

@author: r10p86
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


#https://medium.com/dataseries/visualizing-the-feature-maps-and-filters-by-convolutional-neural-networks-e1462340518e
#https://pub.towardsai.net/tuning-pytorch-hyperparameters-with-optuna-470edcfd4dc
#https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f

#https://www.frontiersin.org/articles/10.3389/fmolb.2019.00044/full


# this function is to be used somewhere else, network is changable
def network():
    hypermodel = hypermodel(CNN1_Team)
    return hypermodel


def calc_outsize(H, W, kern, pad=(0,0), dil=(1,1), stride=(2,2)):
    #H_out = (H + 2 * pad[0] - dil[0] * (kern[0] - 1) - 1) / stride[0] + 1
    #W_out = (W + 2 * pad[1] - dil[1] * (kern[1] - 1) - 1) / stride[1] + 1
    H_out = (H - dil[0] * (kern[0] - 1) - 1) / stride[0] + 1
    W_out = (W - dil[1] * (kern[1] - 1) - 1) / stride[1] + 1
    H_out = np.round(H_out - 0.5, 0)
    W_out = np.round(W_out - 0.5, 0)
    return (H_out, W_out)

# Hyperparameters
def hyperparameters():
        batch_size = 4
        n_epochs   = 16
        kfold = 8
        return batch_size, n_epochs, kfold

class hypermodel():
    def __init__(self, network):
        #        self.layers_dict = {'n_conv_lays': 2, 
        #                            'c1':  8, 'c2': 16, #'c3': 32,
        #                            'ks1': 3, 'ks2': 3, #'ks3': 5
        #                            'p1':  2, 'p2': 3
        #                           }
        self.model = self.init_model(network)
        
    def init_model(self, net):
        model = net()
        return model
#    def get_model_params(self):
#        return self.layers_dict

    def optimizer(self): # need to init model first
        optimizer = optim.Adam(self.model.params)
        return optimizer

    def criterion(self):
        # On some regression problems, the distribution of the target variable 
        # may be mostly Gaussian, but may have outliers, e.g. large or small 
        # values far from the mean value. The Mean Absolute Error, or MAE, loss 
        # is an appropriate loss function in this case as it is more robust to 
        # outliers. It is calculated as the average of the absolute difference 
        # between the actual and predicted values.
        criterion = nn.L1Loss(reduction='sum') # sum over batch instead of averaging
        return criterion
        
# CNN Model
class CNN1_Team(nn.Module):
    
    def __init__(self, n_chan = 2, layers_dict = None):
        super(CNN1_Team, self).__init__()
        #self.layers_dict = layers_dict
        self.conv = nn.Sequential(
                        nn.Conv2d(n_chan, 8, (3,3)),
                        nn.MaxPool2d(2),
                        nn.ReLU(),
                        nn.Conv2d(8, 16, 2),
                        nn.MaxPool2d(3),
                        nn.ReLU(),
                        nn.Dropout(0.4)
                        #nn.Conv2d(16, 32, 3),
                        #nn.ReLU(),
                        #nn.MaxPool2d(3)
                        )
        self.linear = nn.Sequential(
                        nn.Linear(16, 8),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(8, 1)         
                        )
    
    def forward(self, x):
        x = self.conv(x)
        print(x.size)
        x = x.view(x.size(1), -1)
        x = self.linear(x)
        return(x)
        
        


