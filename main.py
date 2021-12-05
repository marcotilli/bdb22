# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:40:02 2021

@author: lordm
"""


# BASEPATH: C:\Users\lordm\Desktop\Work\BigDataBowl_2022
import sys
sys.path.append('./bdb22_github')

from helpers import *
from calc_fieldcontrol import *
from calc_spacevalue import *

year = 2018
# load data
df_games = read_games(year)
df_plays = read_plays(str(year))
df_track = read_track(str(year), set(df_plays.playId))
df_players = read_players()

# get data
train_data, test_data = prep_data()

# init model

# train model

# evaluate model




