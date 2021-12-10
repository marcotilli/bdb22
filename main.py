# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:40:02 2021

@author: lordm
"""

# Import python libraries
# ======================================================================== #
import os
from torch.utils.data import DataLoader

# ======================================================================== #
# base_path: C:\Users\lordm\Desktop\Work\BigDataBowl_2022
base_path = os.getcwd()

# LOAD DATA
# ======================================================================== #
from load_data import load_dataframes
df_plays, df_track, df_players, ids_tuple = load_dataframes(base_path)

# get data
# ======================================================================== #
from build_data import BasicTeamFieldControl#, data_loader_params
from prep_data import get_data
dataset = BasicTeamFieldControl(df_track, df_plays, df_players, ids_tuple, get_data)
#del(df_plays, df_track, df_players)
data_loader = DataLoader(dataset, batch_size = 2, shuffle=True)

#split data
# ======================================================================== #


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

