# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:16:55 2021

@author: lordm
"""

import numpy as np
from load_data      import load_single_play
from calc_fieldcontrol import func_calc_fc_combined, compute_team_frame_control

############################# 
#TODO:
# get_data fertigen
# Dataloader
# Train, Eval, Test Split
##############################


def get_data(ex_gameId, ex_playId, df_track, df_plays, df_players, 
             base_path = None, anim = 'frame'):
    
    # loop over games, plays; below a example play
    #ex_gameId, ex_playId = 2018120208, 3040
    
    ex_track, ex_play = load_single_play(ex_gameId, ex_playId, df_track, df_plays, base_path)
    ex_target = comp_target(ex_play)
    ex_track  = comp_field_influence(ex_play.copy(), ex_track.copy(), anim)
    # ex_track.shape: (142560,8) - due to 22*(120*54)
    #               columns: ['x', 'y', 'influence', 'frameId', 'displayName', 
    #                         'jerseyNumber', 'team', 'control']
    
    # Later, when I want to use player-specific data, I need the infos, but
    # for now (BasicTeamFC_CNN) i only need x,y,control
    # Hence, we can drop all rows with non-unique x&y, control would be redundant 
    team_control = ex_track[['x','y','control']].drop_duplicates(subset = ['x', 'y'])
    # transform control-column to a 2D-numpy array with field-size-shape
    team_control = np.array(team_control.control).reshape(120, 54)
    
    # add 2nd channel: ZOI of Returner (to give model understanding of position)
    returner = df_players[df_players.nflId == int(ex_play.returnerId)].displayName.item()
    ret_zoi = ex_track[ex_track.displayName == returner]['influence']
    ret_zoi = 1 / (1 + np.exp(ret_zoi))
    ret_zoi = np.array(ret_zoi).reshape(120, 54)
    
    ex_fc = np.array([team_control, ret_zoi])
    return ex_fc, ex_target


def comp_field_influence(ex_play, ex_track, anim='frame'):
    if anim == 'frame':
        event = (str(ex_play.specialTeamsPlayType.item()).lower()+ '_received', 
                 'fair_catch')
        ex_track = ex_track.loc[(ex_track.event == event[0])+(ex_track.event == event[1])]
    ex_track_ = func_calc_fc_combined(ex_track)
    if ex_track_.shape[0] < 22: print('ERROR 1')
    h_team = ex_play.homeTeamAbbr.item()
    
    #if anim != 'frame':
    #   temp = np.array([(fId, compute_team_frame_control(frame_track, h_team))
    #                     for (fId, frame_track) in ex_track.groupby(ex_track.frameId)])
    #   frameIds, ex_tracks = ? [list comprehension]
    ex_track_ = compute_team_frame_control(ex_track_, h_team)
        
    return ex_track_


def comp_target(ex_play):
    play_target = ex_play.kickReturnYardage.item()
    return play_target
