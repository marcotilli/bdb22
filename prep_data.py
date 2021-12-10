# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:16:55 2021

@author: lordm
"""

#import numpy as np
#import pandas as pd

from helpers_load import load_single_play
from calc_fieldcontrol import func_calc_fc_combined, compute_team_frame_control
from calc_spacevalue import compute_team_frame_control2 # needs retName

anim = 'frame'



############################# 
#TODO:
# get_data fertigen
# Dataloader
# Train, Eval, Test Split
##############################


def get_data(ex_gameId, ex_playId, df_track, df_plays, BASE_PATH=None):
    
    # loop over games, plays
    #ex_gameId = 2018120208
    #ex_playId = 3040
    
    ex_track, ex_play = load_single_play(ex_gameId, ex_playId, df_track, df_plays, BASE_PATH)
    ex_target = comp_target(ex_play)
    ex_track  = comp_field_influence(ex_play, ex_track)
    # Was möchte ich gerne haben? dim etc
    
    # TODO: aus ex_track einen Model-Input machen
    # ODER: mit prep_data() alle ex_track abspeichern
    
    return ex_fc, ex_target


def comp_field_influence(ex_play, ex_track, anim='frame'):
    if anim == 'frame':
        event = str(ex_play.specialTeamsPlayType.item()).lower()+ '_received'
        ex_track = ex_track.loc[ex_track.event == event]
    ex_track_ = func_calc_fc_combined(ex_track)
    h_team = ex_play.homeTeamAbbr.item()
    
    #if anim != 'frame':
    #temp = np.array([(fId, compute_team_frame_control(frame_track, h_team))
    #                      for (fId, frame_track) in ex_track.groupby(ex_track.frameId)])
    #frameIds, ex_tracks = ? [list comprehension]
    ex_track_ = compute_team_frame_control(ex_track_, h_team)
    
    return ex_track_

def comp_target(ex_play):
    play_target = ex_play.kickReturnYardage.item()
    return play_target



#
# *3) build Dataloader for pytorch (creat already the Tensor for all OR build loader and transf on the fly?)
#
# 4) train, eval, test split (train 2018, eval 2019, test 2020 ?)
#

