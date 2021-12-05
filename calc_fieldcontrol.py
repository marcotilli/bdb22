# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:40:04 2021

@author: lordm
"""

#library(mvtnorm) # np.random.multivariate_normal
library(purrr)


###
# USE:
# 1.) func_calc_fc_combined
# 2.) compute_team_frame_control
#####

def func_calc_fc_combined(ex_track):
    ex_track = compute_distance_from_ball(ex_track)
    ex_track = compute_speed_ratio(ex_track)
    ex_track = compute_next_loc(ex_track)
    ex_track = compute_radius_of_influence(ex_track)
    return ex_track


def compute_team_frame_control(frame_tracking_data, home_team):
    
    team_frame_control = frame_tracking_data[frame_tracking_data.team != 'football']
    #TODO: group_split(displayName) %>%
    # R:      df5 %>% group_split()
    # Python: df5_grouped = df5.groupby(["x1"])
    #         [df5_grouped.get_group(group) for group in df5_grouped.groups]
    team_frame_control_gr = team_frame_control.groupby(team_frame_control.displayName)
    # TODO: map_dfr(., compute_player_zoi) %>%
    team_frame_control_gr = team_frame_control_gr.map(compute_player_zoi)
    # TODO:
        mutate(
            influence = case_when(
                team == home_team ~ -1 * influence,
                TRUE ~ influence
                )
            ) %>%
        group_by(frameId, x, y) %>%
        summarise(control = sum(influence), .groups = "keep") %>%
        mutate(control = 1 / (1 + exp(control)))
  
    return team_frame_control



###############################################################################
#                                 HELPERS                                     #
###############################################################################


# 1. compute player's distance from ball
compute_distance_from_ball <- function(tracking_data) {
  tracking_data <- tracking_data %>%
    inner_join(
      tracking_data %>%
        filter(team == "football") %>%
        select(frameId, ball_x = x, ball_y = y),
      by = "frameId") %>%
    mutate(distance_from_ball = sqrt((x-ball_x)^2 + (y-ball_y)^2)) %>% 
    select(-ball_x, -ball_y)
  return(tracking_data)
}

# 2. compute each player's speed ratio
#    here we're using a max speed of 23 yds/s, 
#    which about lines up with the max speeds seen in 
#    the Next Gen Stats Fastest Player (T. Hill, 2016)
def compute_speed_ratio(tracking_data, s_max = 23.00):
    tracking_data['s_ratio'] <- tracking_data.s / tracking_data.s_max
    return tracking_data

# 3. compute each player's next location
def compute_next_loc(tracking_data, delta_t = 0.50):
    tracking_data['x_next'] = tracking_data.x + tracking_data.v_x * delta_t
    tracking_data['y_next'] = tracking_data.y + tracking_data.v_y * delta_t
    return tracking_data

# 4. compute each player's radius of influence for a given frame
#    here we're using a model that approximates the plot shown in
#    the appendix of Wide Open Spaces. this original function was
#    found by Will Thomson. the modification that I'll make is that
#    I'll add a few parameters to the equation, so we can alter the
#    min/max radius of influence a player can have, as well as the
#    rate at which that radius changes (based on their proximity 
#    to the ball)
def compute_radius_of_influence(tracking_data,
                                min_radius = 4.00,
                                max_radius = 10.00,
                                max_distance_from_ball = 20.00):
    # TODO: Check if correct
    tracking_data['radius_of_influence'] = np.max(max_radius, 
             min_radius + tracking_data.distance_from_ball^3 * 
                 (max_radius-min_radius) / max_distance_from_ball)
    return tracking_data

#################################################
def compute_rotation_matrix(v_theta):
    R = np.array([
            [np.cos(v_theta), -np.sin(v_theta)],
            [np.sin(v_theta),  np.cos(v_theta)]])
    return R

def compute_scaling_matrix(radius_of_influence, s_ratio):
    S = np.array([
            [radius_of_influence * (1 + s_ratio), 0],
            [0, radius_of_influence * (1 - s_ratio)]])
    return S

compute_covariance_matrix <- function(v_theta, radius_of_influence, s_ratio):
    R = compute_rotation_matrix(v_theta)
    S = compute_scaling_matrix(radius_of_influence, s_ratio)
    Sigma <- R %*% S %*% S %*% solve(R)
    return Sigma

# note that this is meant operate on just 1 row of the tracking dataset
compute_player_zoi <- function(player_frame_tracking_data, field_grid = NULL) {
  if (nrow(player_frame_tracking_data)!= 1){
    stop('ERROR: compute_player_zoi only works with 1 row!')
  }
  if(is.null(field_grid)) {
    field_grid <- expand_grid(
      x = seq(0, 120, length.out = 120),
      y = seq(0, 160/3, length.out = 160/3)
    )
  }
  
  frameId_      <- player_frame_tracking_data %>% pull(frameId)
  displayName_  <- player_frame_tracking_data %>% pull(displayName) 
  jerseyNumber_ <- player_frame_tracking_data %>% pull(jerseyNumber) 
  team_         <- player_frame_tracking_data %>% pull(team) 
  
  zoi_center_x_ <- player_frame_tracking_data %>% pull(x_next)
  zoi_center_y_ <- player_frame_tracking_data %>% pull(y_next)
  radius_of_influence_ <- player_frame_tracking_data %>% pull(radius_of_influence)
  v_theta_ <- player_frame_tracking_data %>% pull(v_theta)
  s_ratio_ <- player_frame_tracking_data %>% pull(s_ratio)
  
  mu    <- c(zoi_center_x_, zoi_center_y_)
  Sigma <- compute_covariance_matrix(v_theta_, radius_of_influence_, s_ratio_)
  
  player_zoi <- field_grid %>%
    mutate(
      #influence = mvtnorm::dmvnorm(x = field_grid, mean = mu, sigma = Sigma),
      influence = dmvnorm(x = field_grid, mean = mu, sigma = Sigma),
      influence = influence / max(influence),
      frameId   = frameId_,
      displayName  = displayName_,
      jerseyNumber = jerseyNumber_,
      team = team_
    )
  
  return(player_zoi)
}


