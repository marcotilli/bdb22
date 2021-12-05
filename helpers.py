# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:40:03 2021

@author: lordm
"""

import pandas as pd
import numpy as np


def read_games(year):
  
  df_games = pd.read_csv('data/games.csv')
  df_games = df_games[df_games.season == year]
  df_games = df_games[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']]    
  return df_games


def read_plays(year_string: str):
    
  df_plays = pd.read_csv('data/plays.csv')
  # select games from given year
  play_bools = [str(gId) == year_string for gId in df_plays.gameId]
  df_plays = df_plays[play_bools]
  # only Kickoff or Punts (no Field Goal or XP)
  df_plays = df_plays[df_plays.specialTeamsPlayType == ('Punt' or 'Kickoff')]
  # exclude Muffed, Touchback, Out of Bounds, Downed - since there is no Returner
  df_plays = df_plays[df_plays.specialTeamsResult == ('Fair Catch' or 'Return')]
  # select needed columns
  df_plays = df_plays[['gameId', 'playId', 'possessionTeam', 'specialTeamsPlayType', 
                      'playDescription', 'returnerId', 'absoluteYardlineNumber']]
  return df_plays


def read_track(year_string, playIds_):

    df_track = pd.read_csv('data/tracking'+year_string+'.csv', decimal='.')
    # select only the wanted plays (see read_plays for that)
    play_bools = [pId in playIds_ for pId in df_track.playId]
    df_track = df_track[play_bools]
    # select only useful columns
    df_track = df_track[['gameId', 'playId', 'playDirection', 'x', 'y', 's', 'dir', 
                         'event', 'displayName', 'jerseyNumber', 'frameId', 'team']]
    return df_track


def read_players():
    df_games = read.csv('data/players.csv')[['nflId', 'displayName']]
    return df_games


def extract_single_game(df_games, gId):
  
  ex_game = df_games[df_games.gameId==gId]
  df_colors = fetch_team_colors(ex_game.homeTeamAbbr, ex_game.visitorTeamAbbr)
  ex_game['home_1'] = df_color.home['color1']
  ex_game['home_2'] = df_color.home['color2']
  ex_game['away_1'] = df_color.away['color1']
  ex_game['away_2'] = df_color.away['color2']
  return ex_game


def extract_single_track(df_track, ex_game, gId, pId):
    
    ex_track = df_track[[df_track.gameId==gId & df_track.playId==pId]]
    playdir = ex_track.playDirection[1]
    # flip play, s.t. punt/kick always goes from left to RIGHT
    # x:   120-x ist evtl nicht richtig (v.a. siehe max(df_track$x))
    # dir: dir mod 180 ? -> flips horizontally AND vertically --> also flip y
    if playdir == 'left':
        ex_track.x = 120 - ex_track.x
        ex_track.y = 160/3 - ex_track.y
        ex_track.dir = (ex_track.dir+180) % 360
    ex_track = convert_radiant(ex_track, ex_game)
    return ex_track, playdir


def extract_single_play(df_plays, playdir, gId, pId):
    # if we flipped tracking data -> all plays go from left to right
    ex_play = df_plays [[df_track.gameId==gId & df_track.playId==pId]]          
    if playdir == 'left':
        ex_play.lineofscrimmage = 120 - ex_play.absoluteYardlineNumber
        
    ex_play.drop(columns=['absoluteYardlineNumber'], inplace=True)
    return ex_play


def fetch_team_colors(h_team, a_team):
    team_colors = pd.read_csv('data/team_colors.txt', sep='\t').drop(
                    columns=['color1_family'])
    #colors_url <- "https://raw.githubusercontent.com/asonty/ngs_highlights/master/utils/data/nfl_team_colors.tsv"
    col_bool = [team in (h_team, a_team) for team in team_colors.teams]
    team_colors = team_colors[col_bool]
    team_colors.teams[team_colors.teams == h_team] = 'home'
    team_colors.teams[team_colors.teams == a_team] = 'away'
    return team_colors


def convert_radiant(df, ex_game):
    df['dir'] *= np.pi / 180
    df['v_x'] = np.sin(df['dir']) * df['s']
    df['v_y'] = np.cos(df['dir']) * df['s']
    v_theta = np.arctan(df['v_y'] / df['v_x'])
    df['v_theta'] = [0 if np.isnan(vt) else vt for vt in v_theta]
    team_dict = {'home': ex_game.homeTeamAbbr, 
                 'away': ex_game.visitorTeamAbbr,
                 'football': 'football'}
    df['team'] = df['team'].map(team_dict)
    df[['frameId', 'event', 'team', 'jerseyNumber', 'displayName', 
        'x', 'y', 's', 'v_theta', 'v_x', 'v_y', 'playDirection']]
    return df




