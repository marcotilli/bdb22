#'''
#Calculate Space Value
#'''

source('./calc_fieldcontrol.R')

###
# INPUTS
###
# df_players
# ex_play / ex_play&returnerId
# ex_track
#
#####

#calc_spacevalue <- function(df_players, retId, ex_track, frameId, raw=FALSE){
calc_spacevalue <- function(retName, ex_track, frameId=0, raw=F){
  
  # (1) find Returner and filter tracking data
  #returnerName <- df_players %>% filter(nflId == strtoi(retId))
  ret_track <- ex_track %>% filter(displayName == retName)
  
  if (raw){
    # (1b) filter for frame "kick_received"
    ret_track <- ret_track %>% filter(frameId == frameId)
    # (2) Calculate Bivariate NV for Returner for each frame
    ret_track <- func_calc_fc_combined(ret_track)
  }
  
  # (3) apply SV to whole fied by adapting calc_fieldcontrol/compute_team_frame_control
  space_value_frame <- compute_player_zoi(ret_track) %>% 
                        select(c('frameId', 'x', 'y', 'influence')) %>%
                        mutate_at(vars(influence), list(~ round(., 6)))
  
  return(space_value_frame)
}
  
# (4) apply Space Value to (a) team control 
compute_team_frame_control2 <- function(frame_tracking_data, home_team, retName){
    
    sp_val_frame <- calc_spacevalue(retName, frame_tracking_data)
    
    team_frame_control <- frame_tracking_data %>%
        filter(team != "football") %>%
        group_split(displayName) %>%
        map_dfr(., compute_player_zoi) %>%
        mutate(
          influence = case_when(
            team == home_team ~ -1 * influence,
            TRUE ~ influence
          )
        ) %>%
        group_by(frameId, x, y) %>%
        summarise(control = sum(influence), .groups = "keep") %>%
      mutate(control = 1 / (1 + exp(control)))
  
    team_frame_control$control <-   0.5* (1+team_frame_control$control*sp_val_frame$influence)
    
  return(team_frame_control)
}

#TODO:
# (4) apply Space Value to (b) each players influence 
compute_player_zoi2 <- function(player_frame_tracking_data, field_grid = NULL) {
  
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


