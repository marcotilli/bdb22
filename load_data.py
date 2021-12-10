# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:40:03 2021

@author: lordm
"""

import pandas as pd
import numpy as np

def load_dataframes(base_path):
    years = [2018, 2019, 2020]
    #df_games = read_games(year, base_path)
    print('Read players data ... ', end='\r')
    df_players = read_players(base_path)
    print('Read plays data ... ', end='\r')
    df_plays = read_plays(base_path)
    print('Read tracking data ... ', end='\r')
    df_track = read_track(years, set(df_plays.playId), base_path)
    print('Read play-ids ... ', end='\r')
    gId_pIds = load_playIds(df_plays)
    return df_plays, df_track, df_players, gId_pIds

def load_single_play(ex_gameId, ex_playId, df_track, df_plays, base_path):
    #ex_game  = extract_single_game(df_games, ex_gameId)
    ex_play = df_plays.loc[(df_plays.gameId==ex_gameId) & 
                           (df_plays.playId==ex_playId)].copy() 
    ex_track, playdir = extract_single_track(df_track, ex_play, ex_gameId, ex_playId)
    ex_play = adapt_single_play(ex_play, playdir, base_path)
    return ex_track, ex_play

def load_playIds(df_plays):
    gId_pIds = [(gid, pId) 
                for (gid, group) in df_plays.groupby(df_plays.gameId) 
                for pId in set(group.playId)]
    return gId_pIds

def fetch_team_colors(h_team, a_team, base_path=None):
    
    if base_path is None: base_path = 'C:\\Users\\Tilli\\Desktop\\Privat\\BigDataBowl2022\\'
    team_colors = pd.read_csv(base_path+'\\data\\team_colors.txt', sep='\t').drop(
                    columns=['color1_family'])
    #colors_url <- "https://raw.githubusercontent.com/asonty/ngs_highlights/master/utils/data/nfl_team_colors.tsv"
    team_colors = team_colors[[team in (h_team, a_team) for team in team_colors.teams]]
    team_colors.teams[team_colors.teams == h_team] = 'home'
    team_colors.teams[team_colors.teams == a_team] = 'away'
    return team_colors.set_index('teams')


def read_plays(base_path, years=None):
    # brauche ja nicht nach years filtern, wenn ich eh alle nehme?
    
    df_games = pd.read_csv(base_path+'\\data\\games.csv')
    #df_games = df_games.loc[df_games.season in years][
    #                            ['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']]      
    df_games = df_games[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']]  
    
    #years_str = [str(y) for y in years]
    df_plays = pd.read_csv(base_path+'\\data\\plays.csv')   
    # select games from given year
    #df_plays = df_plays[[str(gId)[:4] in years_str for gId in df_plays.gameId]]
    # only Kickoff or Punts (no Field Goal or XP)
    df_plays = df_plays.loc[(df_plays.specialTeamsPlayType == 'Punt') #|
                            #(df_plays.specialTeamsPlayType == 'Kickoff')
                            ]
    # exclude Touchback, Out of Bounds, Downed - since there is no Returner
    df_plays = df_plays[[a in ('Muffed', 'Fair Catch', 'Return') for a in df_plays.specialTeamsResult]]
    # select needed columns
    df_plays = df_plays[['gameId', 'playId', 'possessionTeam', 'specialTeamsPlayType', 
                         'returnerId','kickReturnYardage', 'absoluteYardlineNumber']]
    df_plays['kickReturnYardage'] = df_plays.kickReturnYardage.fillna(0)
    
    df_plays = df_plays.merge(df_games, on='gameId')
    return df_plays

def read_track(years, playIds_, base_path):
    
    df_track = None
    for year_string in (str(y) for y in years):
        df_temp = pd.read_csv(base_path+'\\data\\tracking'+year_string+'.csv', decimal='.')
        if not df_track is None:
            df_track = df_track.append(df_temp)
        else: 
            df_track = df_temp.copy()
    
    # select only the wanted plays (see read_plays for that)
    df_track = df_track[[pId in playIds_ for pId in df_track.playId]]
    # select only useful columns
    df_track = df_track[['gameId', 'playId', 'playDirection', 'x', 'y', 's', 'dir', 
                         'event', 'displayName', 'jerseyNumber', 'frameId', 'team']]
    return df_track

def read_players(base_path):
    df_games = pd.read_csv(base_path+'\\data\\players.csv')[['nflId', 'displayName']]
    return df_games


def extract_single_track(df_track, ex_play, gId, pId):
    
    #ex_game = df_games[[gameId==gId for gameId in df_games.gameId]].copy()
    ex_track = df_track.loc[(df_track.gameId==gId) & (df_track.playId==pId)].copy()
    playdir = ex_track.playDirection.iloc[0]
    # flip play, s.t. punt/kick always goes from left to RIGHT
    # x:   120-x ist evtl nicht richtig (v.a. siehe max(df_track$x))
    # dir: dir mod 180 ? -> flips horizontally AND vertically --> also flip y
    if playdir == 'left':
        ex_track.x = 120 - ex_track.x
        ex_track.y = 160/3 - ex_track.y
        ex_track.dir = (ex_track.dir+180) % 360
    ex_track = convert_radiant(ex_track, ex_play) # ex_play instead of ex_game
    
    return ex_track, playdir

def adapt_single_play(ex_play, playdir, base_path):
    # flip data -> all plays go from left to right     
    if playdir == 'left': 
        ex_play['lineofscrimmage'] = 120 - ex_play.absoluteYardlineNumber
    else: 
        ex_play['lineofscrimmage'] = ex_play.absoluteYardlineNumber
    ex_play.drop(columns=['absoluteYardlineNumber'], inplace=True)
    
    # add team colors
    team_colors = fetch_team_colors(ex_play.homeTeamAbbr.item(), 
                                  ex_play.visitorTeamAbbr.item(), base_path)
    h = team_colors.loc['home'][['color1', 'color2']]
    a = team_colors.loc['away'][['color1', 'color2']]
    ex_play['home1'], ex_play['home2'] = h[0], h[1] 
    ex_play['away1'], ex_play['away2'] = a[0], a[1] 
    
    return ex_play

def convert_radiant(df, ex_play):
    df['dir'] = df['dir'] * np.pi / 180
    df['v_x'] = np.sin(df['dir']) * df['s']
    df['v_y'] = np.cos(df['dir']) * df['s']
    v_theta = np.arctan(df['v_y'] / df['v_x'])
    df['v_theta'] = [0 if np.isnan(vt) else vt for vt in v_theta]
    team_dict = {'home': ex_play.homeTeamAbbr.item(), 
                 'away': ex_play.visitorTeamAbbr.item(),
                 'football': 'football'}
    df['team'] = df['team'].map(team_dict)
    df = df[['frameId', 'event', 'team', 'jerseyNumber', 'displayName', 
             'x', 'y', 's', 'v_theta', 'v_x', 'v_y']]
    return df.copy()


# ========================================================================== #
#  Archiv
# ========================================================================== #

# def read_games(year, BASE_PATH):
#     df_games = pd.read_csv(BASE_PATH+'\\data\\games.csv')
#     df_games = df_games[df_games.season == year]
#     df_games = df_games[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']]    
#     return df_games

# def extract_single_game(df_games, gId):
#     ex_game = df_games[[gameId==gId for gameId in df_games.gameId]].copy()
#     df_colors = fetch_team_colors(ex_game.homeTeamAbbr.item(), 
#                                   ex_game.visitorTeamAbbr.item())
#     h = df_colors.loc[df_colors.teams == 'home'][['color1', 'color2']]
#     a = df_colors.loc[df_colors.teams == 'home'][['color1', 'color2']]
#     ex_game['home1'] = h['color1']
#     ex_game['home2'] = h['color2']
#     ex_game['away1'] = a['color1']
#     ex_game['away2'] = a['color2']
#     return ex_game