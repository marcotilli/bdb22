#'''
#Calculate Space Value
#'''

###
# INPUTS
###
# retName
# ex_play / ex_play&returnerId
# ex_track
#
#####

import pandas as pd
import numpy as np

from calc_fieldcontrol import func_calc_fc_combined, compute_player_zoi


def calc_spacevalue(retName, ex_track, frameId=None, raw=False):
  
    # (1) find Returner and filter tracking data
    #returnerName = df_players %>% filter(nflId == strtoi(retId))
    ret_track = ex_track.loc[ex_track.displayName == retName].copy()
  
    if raw:
        # (1b) filter for frame "XX_received"
        #TODO: later, need to uncomment this!!!
        #ret_track = ret_track.loc[ret_track.frameId == frameId]
        # (2) Calculate Bivariate NV for Returner for each frame
        ret_track = func_calc_fc_combined(ret_track)
  
    # (3) apply SV to whole fied by adapting calc_fieldcontrol/compute_team_frame_control
    space_value_frame = compute_player_zoi(ret_track)[['frameId', 'x', 'y', 'influence']]
    # use sigmoid to transform influence to [0,1]-interval
    space_value_frame['influence'] = 1 / (1 + np.exp(space_value_frame.influence))
    
    return space_value_frame


# (4) apply Space Value to (a) team control 
def compute_team_frame_control2(frame_track_data, poss_team, retName):
    
    sp_val_frame = calc_spacevalue(retName, frame_track_data)
    
    frm_tr_dat = frame_track_data[frame_track_data.team != 'football'].copy()
    #frm_tr_dat = frm_tr_dat.groupby(frm_tr_dat.displayName)
    #frm_tr_dat = [frm_tr_dat.get_group(g) for g in frm_tr_dat.groups]
    frm_tr_dat = [player_data for (name, player_data) in 
                  frm_tr_dat.groupby(frm_tr_dat.displayName)]
    frm_tr_dat = list(map(compute_player_zoi, frm_tr_dat))
    frm_tr_dat = pd.concat(frm_tr_dat)

    # if player is from punting/kicking team, influence is negative
    frm_tr_dat['influence'] *= (1-2*(frm_tr_dat.team == poss_team))
    #frm_tr_dat.assign(control=frm_tr_dat.groupby(['frameId', 'x', 'y']).influence.agg('sum'))
    frm_tr_dat['control'] = frm_tr_dat.groupby(['x', 'y']).influence.transform('sum')
    
    frm_tr_dat['control'] = 1 / (1 + np.exp(frm_tr_dat['control'])) # control [0,1]
    # apply space value (0 to 1) and transform back to interval [0,1]
    frm_tr_dat['control'] = 0.5* (1 + frm_tr_dat['control'] * sp_val_frame.influence)
    
    return frm_tr_dat
 

#from scipy.stats import multivariate_normal as mv #library(mvtnorm)
#from calc_fieldcontrol import compute_covariance_matrix

#TODO: I think this is the quite the same as applying SV to the team control
# (4) apply Space Value to (b) each players influence 
# def compute_player_zoi_individuel(player_frame_tracking_data, field_grid=None):
  
#     if player_frame_tracking_data.shape[0] != 1:
#         ValueError('ERROR: compute_player_zoi only works with 1 row!')
#     if field_grid is None:
#         x = np.linspace(start=0, stop=120, num=120)
#         y = np.linspace(start=0, stop=160/3, num=160//3+1)
#         field_grid = np.array([(x_, y_) for x_ in x for y_ in y])
#         #field_grid, _ = np.array(np.meshgrid(range(120), range(160//3))) * 0
  
#     frameId_      <- player_frame_tracking_data %>% pull(frameId)
#     displayName_  <- player_frame_tracking_data %>% pull(displayName) 
#     jerseyNumber_ <- player_frame_tracking_data %>% pull(jerseyNumber) 
#     team_         <- player_frame_tracking_data %>% pull(team) 
  
#     zoi_center_x_ <- player_frame_tracking_data %>% pull(x_next)
#     zoi_center_y_ <- player_frame_tracking_data %>% pull(y_next)
#     radius_of_influence_ <- player_frame_tracking_data %>% pull(radius_of_influence)
#     v_theta_ <- player_frame_tracking_data %>% pull(v_theta)
#     s_ratio_ <- player_frame_tracking_data %>% pull(s_ratio)
  
#     mu    <- c(zoi_center_x_, zoi_center_y_)
#     Sigma <- compute_covariance_matrix(v_theta_, radius_of_influence_, s_ratio_)
  
#     player_zoi <- field_grid %>%
#         mutate(
#             #influence = mvtnorm::dmvnorm(x = field_grid, mean = mu, sigma = Sigma),
#             influence = dmvnorm(x = field_grid, mean = mu, sigma = Sigma),
#             influence = influence / max(influence),
#             frameId   = frameId_,
#             displayName  = displayName_,
#             jerseyNumber = jerseyNumber_,
#             team = team_
#             )
  
#     return player_zoi


