# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:40:03 2021

@author: lordm
"""

import pandas as pd
import numpy as np

def load_dataframes(base_path, years=None):
    if years is None:
        years = [2018, 2019, 2020]
    #df_games = read_games(year, base_path)
    print('Read players data ... ', end='\r')
    df_players = read_players(base_path)
    print('Read plays data ... ', end='\r')
    df_plays = read_plays(base_path, years)
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
    
    # delete elements manually where event is missing:
    # season 2020
    [gId_pIds.remove(i) for i in [(2020092004, 144), (2020101800, 932), 
                                  (2020101813, 3391), (2020110106, 1783), 
                                  (2020110807, 762), (2020110810, 272), 
                                  (2020110810, 659), (2020110810, 1270), 
                                  (2020120200, 657), (2020120603, 2737), 
                                  (2020110200, 1572), (2020120610, 3092), 
                                  (2020121700, 826), (2020121900, 1232),
                                  (2020122010, 2634), (2020121900, 1506),
                                  (2020122013, 142), (2020122704, 4087), 
                                  (2020122712, 2995), (2020122712, 3219),
                                  ]]
    [gId_pIds.remove(i) for i in [(2020091310, 701), (2020091307, 121), 
                                  (2020092002, 2683), (2020092711, 2309), 
                                  (2020092711, 2619), (2020101112, 1389), 
                                  (2020101200, 2425), (2020101900, 254),  
                                  (2020112905, 145), (2020122003, 1429) 
                                  ]]
    # season 2018, 2019
    # 2019091500 479, 2019111701 1593, 
    # 2018091600 2116, 2019092205 1605, 
    # 2018100710 2788, 2019102708 962 ,
    # 2019122200 3308, 2018111108 1721, 
    # 2018120208 1565, 2018120903 2565, 
    # 2018090600 3225,2018123011 1435,
    # 2018123014 1602,2018092308 2966,
    # 2019122909 2658,2019112408 3869, 
    # 2018092308 569 ,2018101800 1346, 
    # 2018101100 1387,2019112401 866 ,
    # 2019112405 529 ,2019092907 632 ,
    # 2019100603 479 ,2019122914 2742, 
    # 2018090905 3844,2018112900 2456, 
    # 2019102002 1017, 2018110404 137 ,
    # 2018120209 1259, 2018102102 2179, 
    # 2018093005 3731, 2021010309 3478, 
    # 2019092907 1203, 2018090905 2393, 
    # 2018092000 542 ,2018102104 789 ,
    # 2019090803 617 , 2018112509 3099, 
    # 2018102103 1394, 2019120805 3556,
    # 2018092000 801 ,2019092202 531 ,
    # 2019101700 2334,2021010306 1090, 
    # 2021010311 2781,2018120600 786, 
    # 2018091606 2807, 2019122210 1610, 2018101402 2461
    
    return gId_pIds


def fetch_team_colors(h_team, a_team):
    
    base_path = 'C:\\Users\\Tilli\\Desktop\\Privat\\BigDataBowl2022'
    #if base_path is None: base_path = 'C:\\Users\\lordm\\Desktop\\Work\\BigDataBowl_2022\\'
    team_colors = pd.read_csv(base_path+'\\data\\team_colors.txt', sep='\t').drop(
                    columns=['color1_family'])
    #colors_url <- "https://raw.githubusercontent.com/asonty/ngs_highlights/master/utils/data/nfl_team_colors.tsv"
    team_colors = team_colors[[team in (h_team, a_team) for team in team_colors.teams]]
    team_colors.teams[team_colors.teams == h_team] = 'retTeam'
    team_colors.teams[team_colors.teams == a_team] = 'posTeam'
    return team_colors.set_index('teams')


def read_plays(base_path, years):
    # brauche ja nicht nach years filtern, wenn ich eh alle nehme?
    
    df_games = pd.read_csv(base_path+'\\data\\games.csv')
    if len(years) > 1:
        df_games = df_games[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']]  
        df_plays = pd.read_csv(base_path+'\\data\\plays.csv')
    else: # if 1 year is given
        df_games = df_games.loc[df_games.season == years[0]][
                                  ['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']]      
        df_plays = pd.read_csv(base_path+'\\data\\plays.csv')   
        # select games from given year
        df_plays = df_plays[[str(gId)[:4] == str(years[0]) for gId in df_plays.gameId]]
    
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
    retteam = df_plays.homeTeamAbbr.copy()
    retteam.loc[df_plays.possessionTeam == df_plays.homeTeamAbbr] = df_plays['visitorTeamAbbr'].copy()
    df_plays['returningTeam'] = retteam
    df_plays.drop(columns= ['homeTeamAbbr', 'visitorTeamAbbr'])
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
    df_players = pd.read_csv(base_path+'\\data\\players.csv')[['nflId', 'displayName']].copy()
    
    # manually change some names:
    df_players.loc[df_players.displayName=='Cedrick Wilson', 'displayName'] = 'Ced Wilson'
    
    return df_players


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
    team_colors = fetch_team_colors(ex_play.returningTeam.item(), 
                                    ex_play.possessionTeam.item())
    r = team_colors.loc['retTeam'][['color1', 'color2']] # returningTeam
    p = team_colors.loc['posTeam'][['color1', 'color2']] # possesionTeam
    ex_play['retTeam1'], ex_play['retTeam2'] = r[0], r[1] 
    ex_play['posTeam1'], ex_play['posTeam2'] = p[0], p[1] 
    
    return ex_play

def convert_radiant(df, ex_play):
    df['dir'] = df['dir'] * np.pi / 180
    df['v_x'] = np.sin(df['dir']) * df['s']
    df['v_y'] = np.cos(df['dir']) * df['s']
    v_theta = np.arctan(df['v_y'] / df['v_x'])
    df['v_theta'] = [0 if np.isnan(vt) else vt for vt in v_theta]
    team_dict = {'retTeam': ex_play.returningTeam, 
                 'posTeam': ex_play.possessionTeam,
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