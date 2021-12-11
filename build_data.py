# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 08:43:51 2021

@author: r10p86
"""

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from prep_data import get_data


def build_data_loader(df_track, df_plays, df_players, ids_tuple):
    
    # Data Loader Parameters
    loader_params = {'bs': 1,
                     'shuffle': False, 
                     'num_workers': 1}
    
    dataset = BasicTeamFieldControl(df_track, df_plays, df_players, ids_tuple, get_data)
    train_loader = DataLoader(dataset, batch_size=loader_params['bs'], shuffle=False)
    
    # # Init. DataSet
    # dataset = BasicTeamFieldControl(df_track, df_plays, df_players, ids_tuple, get_data)
    # train_data, test_data = train_test_split(dataset, test_size=0.2, 
    #                                          random_state=42, shuffle=True)
    
    # # Train Eval Test Split
    # k=8
    # splits=KFold(n_splits=k, shuffle=True, random_state=42)
    # foldperf={}
    
    # # The dataloaders handle shuffling, batching, etc...
    # train_loader = DataLoader(train_data, batch_size=loader_params['bs'])
    # #valid_loader = DataLoader(valid_data, batch_size=loader_params['bs'])
    # test_loader  = DataLoader(test_data, batch_size=loader_params['bs'],shuffle=True)
    
    # print("Batches in Train Loader: {}".format(len(train_loader)))
    # print("Batches in Valid Loader: {}".format(len(valid_loader)))
    # print("Batches in Test Loader: {}".format(len(test_loader)))
    
    return train_loader#, test_loader



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







