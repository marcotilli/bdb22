# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:40:42 2021

@author: r10p86
"""

#tqdm
#https://medium.com/@harshit4084/track-your-loop-using-tqdm-7-ways-progress-bars-in-python-make-things-easier-fcbbb9233f24
#plot loss
#https://discuss.pytorch.org/t/visualize-live-graph-of-lose-and-accuracy/27267/6


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import  DataLoader, SubsetRandomSampler

####
# Input
#   dataset
#   splits
#
    
def trainer(splits, dataset, n_epochs, batch_size, device):

    from model import CNN1_Team
    criterion = nn.L1Loss(reduction='sum')
    
    foldperf={}
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        #print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler  = SubsetRandomSampler(val_idx)
        train_loader  = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader   = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
        model = CNN1_Team() # network # init every fold
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
        
        for epoch in tqdm(range(n_epochs), desc='Fold {}'.format(fold + 1)):
            train_loss, train_correct = train_epoch(model,device,train_loader,criterion,optimizer)
            test_loss,  test_correct  = valid_epoch(model,device,test_loader,criterion)
            
            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,n_epochs, train_loss, test_loss, train_acc, test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
        
        foldperf['fold{}'.format(fold+1)] = history  
    
    return model 


###########################################################################


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    
    train_loss, train_correct = 0.0, 0
    model.train()
    
    for tracks, labels in tqdm(dataloader, desc='Train Epoch Loop'):

        tracks, labels = tracks.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(tracks.float())
        loss = loss_fn(output.reshape(-1), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * tracks.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss,train_correct
  
    
  
def valid_epoch(model, device, dataloader, loss_fn):
    
    valid_loss, val_correct = 0.0, 0
    model.eval()
    
    for tracks, labels in tqdm(dataloader, desc='Valid Loop'):

        tracks,labels = tracks.to(device),labels.to(device)
        output = model(tracks.float())
        loss   = loss_fn(output.reshape(-1), labels)
        valid_loss += loss.item()*tracks.size(0)
        scores, predictions = torch.max(output.data,1)
        val_correct += (predictions == labels).sum().item()

    return valid_loss,val_correct