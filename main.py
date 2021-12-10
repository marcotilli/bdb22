# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:40:02 2021

@author: lordm
"""

# BASEPATH: C:\Users\lordm\Desktop\Work\BigDataBowl_2022
import sys, os
BASE_PATH = os.getcwd()
sys.path.append(BASE_PATH+'\\bdb22_github')

from helpers_load import load_dataframes
from prep_data import get_data
from build_data import BasicTeamFieldControl, data_loader_params
from torch.utils.data import Dataloader


years = [2018, 2019, 2020]
# load data
df_plays, df_track, ids_tuple = load_dataframes(years, BASE_PATH)

# get data
dataset = BasicTeamFieldControl(df_track, df_plays, ids_tuple, get_data)
data_loader = Dataloader(dataset, *data_loader_params.values())
#split data


# init model, optimizer
from model import hyperparams


# train model


# evaluate model



# Next:
#   - how to put influence(s) of in a matrix (df_control / space_value_frame)
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

