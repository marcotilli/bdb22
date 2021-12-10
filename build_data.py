# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 08:43:51 2021

@author: r10p86
"""

from torch.utils.data import Dataset

class BasicTeamFieldControl(Dataset):
    def __init__(self, df_track, df_plays, df_players, ids_tuple, get_data):
        self.tracks  = df_track
        self.plays   = df_plays
        self.players = df_players
        self.idxs = ids_tuple
        self.get_data = get_data
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        gId, pId = self.idxs[idx]
        ex_fc, retYards = self.get_data(gId, pId, self.tracks, self.plays, self.players)
                   
        return ex_fc, retYards


# Data Loader
data_loader_params = {'batch_size':  16,
                      'shuffle':     True, 
                      'num_workers': 2}


# Train Eval Test Split
