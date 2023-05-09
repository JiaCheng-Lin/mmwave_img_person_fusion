# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter




class cam_mmw(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

## regression model network
class bbox2MMW_Model(nn.Module):
    def __init__(self, input_dim):
        super(bbox2MMW_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            
            nn.Linear(8, 16),
            nn.ReLU(),
            
            nn.Linear(16, 32),
            nn.ReLU(),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            
            nn.Linear(8, 2)

        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 100000,     # Number of epochs.            
    'batch_size': 512, 
    'learning_rate': 1e-2,    # original: 1e-5 
    'early_stop': 2000,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': r'C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\cal_tranform_matrix\regression_models/'  # Your model will be saved here.
}

def get_regression_model(model_name):
    model = bbox2MMW_Model(input_dim=4).to('cuda')
    model.load_state_dict(torch.load(config['save_path']+model_name))

    return model.eval()


def predict(model, data): # data: np array
    if data.size != 0:
        test_dataset =  cam_mmw(data[:, :-1]) # np.array([[104., 392.28721168, 105.26521565, 294.57442337]])
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
        
        preds = []
        for x in test_loader: # tqdm(test_loader):
            x = x.to('cuda')                        
            with torch.no_grad():                   
                pred = model(x)                     
                preds.append(pred.detach().cpu())   
        preds = torch.cat(preds, dim=0).numpy() 

        return preds
    return []

# if __name__ == '__main__':

    # test_dataset =  bbox2MMW_Model(np.array([[104.        , 245.        , 105.26521565, 294.57442337]]))
    # test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    # model = bbox2MMW_Model(input_dim=2).to(device)
    # model.load_state_dict(torch.load(config['save_path']))
    # preds = predict(test_loader, model, device) 
    # print(preds)