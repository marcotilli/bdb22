# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:40:02 2021

@author: lordm
"""

# Import python libraries
# ======================================================================== #
import os, sys
from tqdm import tqdm
# ======================================================================== #
# base_path: C:\Users\lordm\Desktop\Work\BigDataBowl_2022
base_path = os.getcwd()
sys.path.append('.\\bdb22_github')

# ======================================================================== #
from load_data import load_dataframes
years = [2020]
df_plays, df_track, df_players, ids_tuples = load_dataframes(base_path, years)

# get Dataloader (init Dataset and split data)
# ======================================================================== #
from build_data import build_data_loader

train_loader = build_data_loader(df_track, df_plays, df_players, ids_tuples)
for _ in tqdm(train_loader):
    pass

# init model, optimizer
# ======================================================================== #
from model import hyperparams


# ======================================================================== #
# train model


# ======================================================================== #
# evaluate model



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

