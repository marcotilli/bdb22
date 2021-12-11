# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 08:43:51 2021

@author: r10p86
"""

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from prep_data import get_data


def build_data_loader(df_track, df_plays, df_players, ids_tuples):
    
    # Data Loader Parameters
    #loader_params = {'bs': 1,
    #                 'shuffle': False, 
    #                 'num_workers': 1}
    
    # load data
    dataset = BasicTeamFieldControl(df_track, df_plays, df_players, ids_tuples, get_data)
    # split into train and test data
    train_dataset, test_dataset = train_test_split(dataset, test_size = 0.25)
    # init test data loader
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print("Examples in Train Data: {}".format(len(train_dataset)))
    print("Examples in Test Loader: {}".format(len(test_loader)))
    
    return train_dataset, test_loader



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



# for TESTING:
# train_loader = build_data_loader(df_track, df_plays, df_players, ids_tuples)
# for test in tqdm(train_loader):
#     test_data, test_target = test
#     break


