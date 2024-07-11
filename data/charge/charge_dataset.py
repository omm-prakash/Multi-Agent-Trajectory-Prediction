import numpy as np
import pandas as pd

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--data-path', type=str, default=os.path.join('data', 'charge_simulation'), help='Directory with the dataset.')
# parser.add_argument('--dataset', type=str, default='charged', help='name_format on which data is needed.')
# parser.add_argument('--n-charges', type=int, default='5', help='No. of charges in the simulation.')

# args = parser.parse_args()

class ChargedParticleData(Dataset):
    def __init__(self, mode, directory, n_particles, name_format, scale_data=True, custom_range=True) -> None:
        super().__init__()
        self.mode = mode
        self.directory = directory
        self.n_particles = n_particles
        self.name_format = name_format
        self.scale_data = scale_data
        self.custom_range = custom_range

        (self.charge, self.location, self.velocity) = self.load_data()
        self.dataset = self.transform_data(self.charge, self.location, self.velocity) # shape: (sample, time, n_particles, 5)
        # self.dataset = self.dataset[0, :, :, :].unsqueeze(0)

    def load_data(self): 
        charge = np.load(os.path.join(self.directory, self.name_format.format(data='charges', mode=self.mode, n=self.n_particles))) # shape: (sample, n_particles, 1)
        location = np.load(os.path.join(self.directory, self.name_format.format(data='loc', mode=self.mode, n=self.n_particles))) # shape: (sample, time, 2, n_particles)
        velocity = np.load(os.path.join(self.directory, self.name_format.format(data='vel', mode=self.mode, n=self.n_particles))) # shape: (sample, time, 2, n_particles)

        charge = torch.from_numpy(charge) # shape: (sample, n_particles, 1)
        location = torch.from_numpy(location) # shape: (sample, time, 2, n_particles)
        velocity = torch.from_numpy(velocity) # shape: (sample, time, 2, n_particles)
        return charge, location, velocity

    def transform_data(self, charge, location, velocity):
        time = velocity.size(1)
        location_ = location.transpose(-1, -2) # shape: (sample, time, n_particles, 2)
        velocity_ = velocity.transpose(-1, -2) # shape: (sample, time, n_particles, 2)
        charge_ = charge.unsqueeze(1).repeat(1,time,1,1) # shape: (sample, time, n_particles, 1)
        
        data = torch.concat([charge_, location_, velocity_], dim=-1) # shape: (sample, time, n_particles, 5)
        self.n_features = data.size(-1)
        self.n_particles = data.size(-2)
        return data # shape: (sample, time, n_particles, 5)

    def __len__(self):
        return self.dataset.size(0)
    
    def __getitem__(self, index):
        data = self.dataset[index] # shape: (time, n_particles, 5)
        if self.scale_data:
            data_ = data.reshape(-1, data.size(-1))
            if self.custom_range:
                min_ = torch.tensor([-1, -5, -5, -7, -7])
                max_ = torch.tensor([1, 5, 5, 7, 7])
            else:
                min_ = data_.min(dim=0)[0]
                max_ = data_.max(dim=0)[0]
            data_ = (data_-min_)/(max_-min_)
            # print(data_.min(dim=0)[0], data_.max(dim=0)[0])
            if torch.isnan(data_).any():
                print('nan value recieved in index', index)
            # print(data_.size(), data.size(), index)
            data_ = data_.view(data.size())
            # data_[:, :, 0] = data[:, :, 0]
            data = data_
        return {
            'data': data, # shape: (1, time, n_particles, 5)
            # 'location': self.location[index], # shape: (1, time, 2, n_particles)
            # 'velocity': self.velocity[index], # shape: (1, time, 2, n_particles)
            # 'charge': self.charge[index] # shape: (1, n_particles, 1)
        }    
