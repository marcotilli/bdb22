# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 08:43:51 2021

@author: r10p86
"""

from torch.utils.data import Dataset, DataLoader, random_split
from prep_data import get_data


def build_data_loader(df_track, df_plays, df_players, ids_tuples):
    
    # Data Loader Parameters
    #loader_params = {'bs': 1,
    #                 'shuffle': False, 
    #                 'num_workers': 1}
    
    # load data
    print('Create Dataset ...', end='\r')
    dataset = BasicTeamFieldControl(df_track, df_plays, df_players, ids_tuples, get_data)
    # split into train and test data
    print('Split Data ... ', end='\r')
    #train_dataset, test_dataset = train_test_split(dataset, test_size = 0.25)
    lengths = [int(len(dataset)*0.80)+1, int(len(dataset)*0.20)]
    train_dataset, test_dataset = random_split(dataset, lengths)
    
    # init test data loader
    print('Creat DataLoader ... ', end='\r')
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


class PreparedDS(Dataset):
    def __init__(self):
        df = pd.read_csv('../input/cnn1-traindata2020/data2020.csv', index_col=0)
        self.targets = df['y_hat'].to_numpy()
        ex_team = list(df['team'])
        ex_ret = list(df['ret'])
        ex_team = np.array([[float(i) for i in row[1:-1].split(',')] for row in list(df['team'])])
        ex_ret  = np.array([[float(i) for i in row[1:-1].split(',')] for row in list(df['ret'] )])
        self.ex_track = np.stack((ex_team.reshape(-1, 120, 54), ex_ret.reshape(-1, 120, 54)), axis=1)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.ex_track[idx], self.targets[idx]

# for TESTING:
# train_loader = build_data_loader(df_track, df_plays, df_players, ids_tuples)
# for test in tqdm(train_loader):
#     test_data, test_target = test
#     break


