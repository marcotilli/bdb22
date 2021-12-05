# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:16:55 2021

@author: lordm
"""

import numpy as np
import pandas as pd

from helpers import *
from calc_fieldcontrol import *
from calc_spacevalue import *


ex_gameId = 2018120208
ex_playId = 3040

ex_game  = adapt_single_game(df_games, ex_gameId)
ex_track, playdir = adapt_single_track(df_track, ex_game, ex_gameId, ex_playId)
ex_play  = adapt_single_play(df_plays, playdir, ex_gameId, ex_playId)


