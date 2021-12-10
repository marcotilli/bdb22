# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:40:04 2021

@author: lordm
"""

from scipy.stats import multivariate_normal as mv #library(mvtnorm)
import pandas as pd
import numpy as np


###
# USE:
# 1.) func_calc_fc_combined
# 2.) compute_team_frame_control
#####

def func_calc_fc_combined(ex_track):
    
    ex_track = comp_distance_from_ball(ex_track)
    ex_track = comp_speed_ratio(ex_track)
    ex_track = comp_next_loc(ex_track)
    ex_track = comp_radius_of_influence(ex_track)
    
    return ex_track


def compute_team_frame_control(frame_track_data, home_team):
    #frame_track_data, home_team = ex_track_, h_team
    
    frm_tr_dat = frame_track_data[frame_track_data.team != 'football'].copy()
    #frm_tr_dat = frm_tr_dat.groupby(frm_tr_dat.displayName)
    #frm_tr_dat = [frm_tr_dat.get_group(g) for g in frm_tr_dat.groups]
    if frm_tr_dat.shape[0] < 22: print('ERROR 1')
    frm_tr_dat = [player_data for (name, player_data) in 
                  frm_tr_dat.groupby(frm_tr_dat.displayName)]
    frm_tr_dat = list(map(compute_player_zoi, frm_tr_dat))
    frm_tr_dat = pd.concat(frm_tr_dat)
    
    # if player is from home_team, influence is negative
    frm_tr_dat['influence'] *= (1-2*(frm_tr_dat.team == home_team))
    #frm_tr_dat.assign(control=frm_tr_dat.groupby(['frameId', 'x', 'y']).influence.agg('sum'))
    frm_tr_dat['control'] = frm_tr_dat.groupby(['x', 'y']).influence.transform('sum')
    frm_tr_dat['control'] = 1 / (1 + np.exp(frm_tr_dat['control'])) # control [0,1]
    
    return frm_tr_dat


###############################################################################
#                                 HELPERS                                     #
###############################################################################


# 1. compute player's distance from ball
def comp_distance_from_ball(tracking_data): 
    track_ball = tracking_data[tracking_data.team == 'football'][['frameId', 'x', 'y']]
    track_ball.columns = ['frameId', 'ball_x', 'ball_y']
    tracking_data = tracking_data.merge(track_ball, how = 'inner', on = 'frameId')
    tracking_data['distance_from_ball'] = np.sqrt((tracking_data.x - tracking_data.ball_x)**2 
                                             + (tracking_data.y - tracking_data.ball_y)**2)
    tracking_data.drop(columns=['ball_x', 'ball_y'], inplace=True)
    return tracking_data

# 2. compute each player's speed ratio
#    here we're using a max speed of 23 yds/s, 
#    which about lines up with the max speeds seen in 
#    the Next Gen Stats Fastest Player (T. Hill, 2016)
def comp_speed_ratio(tracking_data, s_max = 23.00):
    tracking_data['s_ratio'] = tracking_data.s / s_max
    return tracking_data

# 3. compute each player's next location
def comp_next_loc(tracking_data, delta_t = 0.50):
    tracking_data['x_next'] = tracking_data.x + tracking_data.v_x * delta_t
    tracking_data['y_next'] = tracking_data.y + tracking_data.v_y * delta_t
    return tracking_data

# 4. compute each player's radius of influence for a given frame. 
#    Here we're using a model that approximates the plot shown in the appendix
#    of "Wide Open Spaces". This original function was found by Will Thomson. 
#    The modification that I'll make is that I'll add a few parameters to the 
#    equation, so we can alter the min/max radius of influence a player can have,
#    as well as the rate at which that radius changes (based on proximity to ball) 
def comp_radius_of_influence(tracking_data, min_rad = 4.00, max_rad = 10.00, 
                                max_dist_ball = 20.00):
    tracking_data['radius_of_influence'] = np.minimum(
        max_rad, min_rad + (tracking_data.distance_from_ball**3) * (max_rad-min_rad)/max_dist_ball)
    return tracking_data

#################################################

def comp_rotation_matrix(v_theta):
    R = np.array([
            [np.cos(v_theta), -np.sin(v_theta)],
            [np.sin(v_theta),  np.cos(v_theta)]])
    return R

def comp_scaling_matrix(radius_of_influence, s_ratio):
    S = np.array([
            [radius_of_influence * (1 + s_ratio), 0],
            [0, radius_of_influence * (1 - s_ratio)]])
    return S

def comp_covariance_matrix(v_theta, radius_of_influence, s_ratio):
    R = comp_rotation_matrix(v_theta)
    S = comp_scaling_matrix(radius_of_influence, s_ratio)
    Sigma = np.matmul(np.matmul(np.matmul(R, S), S), np.linalg.inv(R))
    return Sigma

# note that this is meant operate on just 1 row of the tracking dataset
def compute_player_zoi(pftr_data, field_grid = None):
    #pftr_data = player_frame_tracking_data.copy()
    
    if pftr_data.shape[0] != 1:
        ValueError('ERROR: compute_player_zoi only works with 1 row!')
    if field_grid is None:
        x = np.linspace(start=0, stop=120, num=120)
        y = np.linspace(start=0, stop=160/3, num=160//3+1)
        field_grid = np.array([(x_, y_) for x_ in x for y_ in y])
        #field_grid, _ = np.array(np.meshgrid(range(120), range(160//3))) * 0
  
    zoi_center_x_, zoi_center_y_ = pftr_data.x_next, pftr_data.y_next
    mu    = np.array((zoi_center_x_, zoi_center_y_)).squeeze()
    sigma = comp_covariance_matrix(pftr_data.v_theta.item(), 
                                   pftr_data.radius_of_influence.item(), 
                                   pftr_data.s_ratio.item())
  
    influence = mv.pdf(x = field_grid, mean = mu, cov = sigma)
    influence = influence / np.max(influence)
    frameId = np.repeat(pftr_data.frameId.item(), len(influence))
    team    = np.repeat(pftr_data.team.item(),    len(influence))
    displayName  = np.repeat(pftr_data.displayName.item(), len(influence))
    jerseyNumber = np.repeat(pftr_data.jerseyNumber.item(), len(influence))      
    
    player_zoi = pd.DataFrame({'x': field_grid[:,0], 
                               'y': field_grid[:,1], 
                               'influence': influence,
                               'frameId': frameId,
                               'displayName': displayName, 
                               'jerseyNumber': jerseyNumber,
                               'team': team})
    
    return player_zoi


