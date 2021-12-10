# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 08:43:51 2021

@author: r10p86
"""

from torch.utils.data import Dataset


class BasicTeamFieldControl(Dataset):
    def __init__(self, df_track, df_plays, ids_tuple, 
                         data_function, additional_fct = None):
        self.df_track = df_track
        self.df_plays = df_plays
        self.idxs = ids_tuple
        self.transform1 = data_function
        self.transform2 = additional_fct
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        gId, pId = self.idxs[idx]
        ex_fc, ex_game = self.transform(gId, pId, self.df_track, self.df_plays)
        
        if not self.transform2 is None:
            ex_fc_channel2 = ... Team-Control SpaceValue
            ex_fc = [ex_fc, ex_fc_channel2]
            
        return ex_fc, ex_game


# Data Loader
data_loader_params = {'batch_size':  16,
                      'shuffle':     True, 
                      'num_workers': 2}


# Train Eval Test Split
