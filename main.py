# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:40:02 2021

@author: lordm
"""

# Import python libraries
# ======================================================================== #
import os, sys
import torch
import pandas as pd
#from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# Init overall stuff
# ======================================================================== #
# base_path: C:\Users\lordm\Desktop\Work\BigDataBowl_2022
base_path = os.getcwd()
sys.path.append('./bdb22_github')
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LOAD DATA
# ======================================================================== #
from load_data import load_dataframes
years = None # take every year
df_plays, df_track, df_players, ids_tuples = load_dataframes(base_path, years)

# GET DATA
# ======================================================================== #
from build_data import build_data_loader
dataset, testloader = build_data_loader(df_track, df_plays, df_players, ids_tuples)

from tqdm import tqdm
trainloader = DataLoader(dataset, batch_size=1, shuffle=False)
data_csv = []
for data, target in tqdm(trainloader):
    if torch.sum(data).item() > 0:
        data_team = data.squeeze()[0].view(-1)
        data_ret  = data.squeeze()[1].view(-1)
        data_csv.append([target.item(), data_team.tolist(), data_ret.tolist()])
    
dataframe = pd.DataFrame(data_csv, columns=['target', 'fc_team', 'fc_ret']) 
dataframe.to_csv('traindata.csv')



# INIT HYPERPARAMETERS, K-FOLD
# ======================================================================== #
torch.manual_seed(42)
from model import hyperparameters
n_epochs, batch_size, kfold = hyperparameters()
splits = KFold(n_splits = kfold, shuffle = True, random_state = 42)


# TRAIN MODEL
# ======================================================================== #
from train_eval import trainer
model = trainer(splits, dataset, n_epochs, batch_size, device)
torch.save(model,'k_cross_CNN.pt')

# TEST MODEL
# ======================================================================== #



















# Next:
#   - use matrix for CNN (team or each players, resp.) 
#        a) SpaceValued-TeamControl: 2 Channel (Team Non-weighted, Returner-SV)
#              1. nur Punts, frame zu punt_receeived; Goal: RetYards
#              ?2. Punts + Kicks, wie 1.
#              3. mit LSTM und über mehrere Frames 
#                  (zB AvgTimeDiff Received&FirstContact oder Received&Tackle)
#        b) XGBoost: 
#               each player: mu, sigma, influence, dir, o, v, ...  
#        c) SpaceValued Players
#              1. nur Punts, Input: 23 Channels (each Player, + non-weighted TeamControl) 
#                  frame zu punt_receeived
#              ?2. Punts und Kicks, wie 1.
#              3. mit LSTM und über mehrere Frames 
#                  (zB AvgTimeDiff Received&FirstContact oder Received&Tackle)
#        d) wie xRushYards, aber mit mu, sigma von ZOI
#
#   - Orientation instead/with Direction


###################################
# Literatur
#https://www.frontiersin.org/articles/10.3389/fdata.2019.00014/full
#https://www.researchgate.net/publication/343122623_Deep_soccer_analytics_learning_an_action-value_function_for_evaluating_soccer_players

