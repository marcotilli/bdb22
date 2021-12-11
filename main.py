# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:40:02 2021

@author: lordm
"""

# Import python libraries
# ======================================================================== #
import os, sys
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# Init overall stuff
# ======================================================================== #
# base_path: C:\Users\lordm\Desktop\Work\BigDataBowl_2022
base_path = os.getcwd()
sys.path.append('./bdb22')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LOAD DATA
# ======================================================================== #
from load_data import load_dataframes
years = [2020]
df_plays, df_track, df_players, ids_tuples = load_dataframes(base_path, years)

# GET DATA
# ======================================================================== #
from build_data import build_data_loader
dataset, testloader = build_data_loader(df_track, df_plays, df_players, ids_tuples)

#trainloader = DataLoader(dataset, batch_size=1, shuffle=False)
#for tester in tqdm(trainloader):
#    tester_data, tester_target = tester
#    break

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

