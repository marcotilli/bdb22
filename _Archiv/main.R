library(tidyverse)
library(purrr)
library(ggplot2)
library(gganimate)

source('./helpers.R')
source('./bdb_animate_plot_field.R')
source('./calc_fieldcontrol.R')
source('./calc_spacevalue.R')

year <- 2018
df_games <- read_games(year)
df_plays <- read_plays(year)
df_track <- read_track(year, unique(c(df_plays$playId)))
df_players <- read_players()

# take one example play and extract data
# Kickoff Medium Gain:  2018090905 3524
# Kickoff TD:           2018102102 1452
# Punt Medium Gain:     2018101402 1357
# Punt Fair catch:      2018101406 2021
# Punt TD:              2018120208 3040
ex_gameId <- 2018120208
ex_playId <- 3040

ex_game  <- adapt_single_game(df_games, ex_gameId)
ex_track <- adapt_single_track_flip(df_track, ex_game, ex_gameId, ex_playId)
ex_play  <- adapt_single_play_flip(df_plays, ex_track$playDirection[1], ex_gameId, ex_playId)
ex_track <- ex_track %>% select(-c('playDirection'))
#rm(df_track, df_plays, df_games)

method = 'field_control_basic'
#method = 'field_control_spacevalue'
event_ = c(paste('punt', 'received', sep='_'),
           paste('kick', 'received', sep='_'),
           'fair_catch')

if (method == 'animate'){
  play_anim <- bdb_animate_plot_field(ex_game, ex_play, ex_track, '', method = method)
  
} else if (method == 'frame' || method == 'field_control_basic'){
    ex_track_ <- func_calc_fc_combined(ex_track)
    if (method == 'frame'){
      ex_track_ <- ex_track_ %>% filter(event %in% event_)
    }
    df_control  <- ex_track_ %>%
        filter(team != "football") %>%
        group_split(frameId) %>%
        map_dfr(., compute_team_frame_control, ex_game$homeTeamAbbr)

    play_anim <- bdb_animate_plot_field(ex_game, ex_play, ex_track_, df_control, method)
} else if (method == 'field_control_spacevalue'){
    
    retName  <- df_players %>% filter(nflId == strtoi(ex_play$returnerId))
    ex_track <- func_calc_fc_combined(ex_track)
    df_control  <- ex_track %>%
        filter(team != "football") %>%
        group_split(frameId) %>%
        map_dfr(., compute_team_frame_control2, ex_game$homeTeamAbbr, retName$displayName)
    
    play_anim <- bdb_animate_plot_field(ex_game, ex_play, ex_track, df_control, method)
}

play_anim


# Next:
#   - how to put influence(s) of in a matrix (df_control / space_value_frame)
#   - transfer to python (So)
#   - use matrix for CNN (team or each players, resp.) (Mo Start)
#        a) SpaceValued-TeamControl (aa non-weighted ab SV-weighted)
#              1. nur Punts, Input: 1 Channel, frame zu punt_receeived; Goal: RetYards
#              2. Punts + Kicks, wie 1.
#              3. mit LSTM und über mehrere Frames 
#                  (zB AvgTimeDiff Received&FirstContact oder Received&Tackle)
#        b) SpaceValued Players
#              1. nur Punts, Input: 23 Channels (each Player, + non-weighted TeamControl) 
#                  frame zu punt_receeived
#              2. Punts und Kicks, wie 1.
#              3. mit LSTM und über mehrere Frames 
#                  (zB AvgTimeDiff Received&FirstContact oder Received&Tackle)



###################################
# Literatur
#https://www.frontiersin.org/articles/10.3389/fdata.2019.00014/full
#https://www.researchgate.net/publication/343122623_Deep_soccer_analytics_learning_an_action-value_function_for_evaluating_soccer_players