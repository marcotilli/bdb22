library(tidyverse)

read_games <- function(year){
  
  # select basic
  df_games <- read.csv('./data/games.csv') %>%
    filter(season == year) %>%
    select(gameId, homeTeamAbbr, visitorTeamAbbr)
  
  # select specific
  #df_games <- filter()
  return(df_games)
}

read_plays <- function(year){
  st_results<- c('Fair Catch', 'Muffed', 'Return') 
  # exclude Touchback, Out of Bounds, Downed - since there is no Returner
  
  #TODO: exclude Onside Kicks
  
  df_plays <- read.csv('./data/plays.csv') %>%
    filter(startsWith(as.character(gameId), as.character(year))) %>%
    # only Kickoff or Punts (no Field Goal or XP)
    filter(specialTeamsPlayType %in% c('Punt', 'Kickoff')) %>%
    # only "finished" Kicks (no blocked or Non-Special-Teams Result)
    filter(specialTeamsResult %in% st_results) %>%
    # select only useful columns
    select(gameId, playId, possessionTeam, specialTeamsPlayType, playDescription, returnerId, absoluteYardlineNumber)
  
  return(df_plays)
}

read_track <- function(year, playId_){

  df_track <- read.csv(paste('./data/tracking', as.character(year), '.csv', sep=''), sep=',', dec='.') %>%
    filter(playId %in% playId_) %>%
    # select only useful columns
    select(gameId, playId, playDirection, x, y, s, dir, event, displayName, jerseyNumber, frameId, team)
  
  return(df_track)
}

read_players <- function(){
  
  df_games <- read.csv('./data/players.csv') %>% 
    select(nflId, displayName)
  
  return(df_games)
}

adapt_single_game <- function(df_games, gId){
  
  ex_game <- df_games %>% filter(gameId==gId)
  df_colors <- fetch_team_colors(ex_game$homeTeamAbbr, ex_game$visitorTeamAbbr)
  ex_game <- ex_game %>%
              mutate(
                home_1 = df_colors$home_1,
                home_2 = df_colors$home_2,
                away_1 = df_colors$away_1,
                away_2 = df_colors$away_2
              )
  return(ex_game)
}

adapt_single_track <- function(df_track, ex_game, gId, pId){
  ex_track <- df_track %>% 
                  filter(gameId==gId, playId==pId) %>%
                  select(-c('gameId', 'playId'))
   ex_track = convert_radiant(ex_track, ex_game)
  return(ex_track)
}
adapt_single_track_flip <- function(df_track, ex_game, gId, pId){
  ex_track <- df_track %>% 
                  filter(gameId==gId, playId==pId) %>%
                  mutate(x_ = x, y_ = y, dir_ = dir) %>%
                  select(-c('gameId', 'playId')) %>%
  # flip play, s.t. punt/kick always goes from left to RIGHT
    # x:   ? 120-x ist nicht richtig glaube ich? (v.a. siehe max(df_track$x))
    # dir: dir mod 180 ? -> flips horizontally AND vertically --> also flip y
                  mutate(x = ifelse(playDirection == 'right', x_, 120 - x_),
                         y = ifelse(playDirection == 'right', y_, 160/3 - y_),
                         dir = ifelse(playDirection == 'right', dir_, (dir+180)%%360)) #%>%
                  #select(-c(x_, y_, dir_))
  ex_track = convert_radiant(ex_track, ex_game)
  return(ex_track)
}

adapt_single_play <- function(df_plays, playdir, gId, pId){
  ex_play <- df_plays %>% 
              filter(gameId==gId, playId==pId) %>%
              mutate(line_of_scrimmage = absoluteYardlineNumber) %>%
              select(-c(absoluteYardlineNumber))
  return(ex_play)
}
# if we flipped tracking data -> all plays go from left to right
adapt_single_play_flip <- function(df_plays, playdir, gId, pId){
  ex_play <- df_plays %>% 
              filter(gameId==gId, playId==pId) %>%
              mutate(line_of_scrimmage = ifelse(
                        playdir == 'right',
                        absoluteYardlineNumber,
                        120 - absoluteYardlineNumber)) %>%
              select(-c(absoluteYardlineNumber))
  
  return(ex_play)
}


fetch_team_colors <- function(h_team, a_team) {
  team_colors = read.csv('./data/team_colors.txt', sep='\t')
  #colors_url <- "https://raw.githubusercontent.com/asonty/ngs_highlights/master/utils/data/nfl_team_colors.tsv"
  
  h_team_col <- team_colors %>% filter(teams == h_team) 
  a_team_col <- team_colors %>% filter(teams == a_team)
  df_colors <- tibble(
    home_1 = h_team_col$color1, home_2 = h_team_col$color2, away_1 = a_team_col$color1, away_2 = a_team_col$color2
  )
  return(df_colors)
}

convert_radiant <- function(dftrack, ex_game){
  
  dftrack <- dftrack %>%
      mutate(
      dir_rad = dir * pi / 180,
      v_x = sin(dir_rad) * s,
      v_y = cos(dir_rad) * s,
      v_theta = atan(v_y / v_x),
      v_theta = ifelse(is.nan(v_theta), 0, v_theta),
      team_name = case_when(
        team == "home" ~ ex_game$homeTeamAbbr,
        team == "away" ~ ex_game$visitorTeamAbbr,
        TRUE ~ team,
      )) %>%
      select(frameId, event, team = team_name, jerseyNumber, displayName, 
             x, y, s, v_theta, v_x, v_y, playDirection)
  return(dftrack)
}




