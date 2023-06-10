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

## for regression model prediction. (mmwave pt project to image)
from joblib import dump, load # save and load model.
from sklearn.model_selection import train_test_split # split data to train&test
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures # Polynomial
from sklearn.pipeline import make_pipeline


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

## bbox to MMW(pixel to position) regression model network
class bbox2MMW_Model(nn.Module):
    def __init__(self, input_dim):
        super(bbox2MMW_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(

            # # ### 20230528
            # nn.Linear(input_dim, 8),
            # nn.ReLU(),
            
            # nn.Linear(8, 16),
            # nn.ReLU(),
            
            # nn.Linear(16, 32),
            # nn.ReLU(),
            
            # nn.Linear(32, 16),
            # nn.ReLU(),
            
            # nn.Linear(16, 8),
            # nn.ReLU(),
            
            # nn.Linear(8, 2)


            ### 20230530 &&  20230602
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            
            nn.Linear(8, 16),
            nn.ReLU(),
            
            nn.Linear(16, 32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
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
        
# MMW to bbox(position to pixel) regression model network
class MMW2bbox_Model(nn.Module):
    def __init__(self, input_dim):
        super(MMW2bbox_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            ### 20230528
            # nn.Linear(input_dim, 8),
            # nn.ReLU(),
            
            # nn.Linear(8, 16),
            # nn.ReLU(),
            
            # nn.Linear(16, 32),
            # nn.ReLU(),

            # nn.Linear(32, 8),
            # nn.ReLU(),
            
            # nn.Linear(8, 2)


            ### 20230530 && 20230602
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 2)
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

def get_bbox2MMW_regression_model(model_name, input_dim=4):
    model = bbox2MMW_Model(input_dim=input_dim).to('cuda')
    model.load_state_dict(torch.load(config['save_path']+model_name))

    return model.eval()

def get_MMW2bbox_regression_model(model_name, input_dim=6):
    model = MMW2bbox_Model(input_dim=input_dim).to('cuda')
    model.load_state_dict(torch.load(config['save_path']+model_name))

    return model.eval()

def predict_pos(model, BBOXs): # data: np array, [[bottom_x, bottom_y, w, h, C_ID], ...]
    data = np.array([[a.bottom_x, a.bottom_y] for a in BBOXs])
    if data.size != 0:
        test_dataset =  cam_mmw(data) 
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
        
        preds = []
        for x in test_loader: # tqdm(test_loader):
            x = x.to('cuda')                        
            with torch.no_grad():                   
                pred = model(x)                     
                preds.append(pred.detach().cpu())   
        preds = torch.cat(preds, dim=0).numpy() 
        
        for i, (Xr, Yr) in enumerate(preds):
            BBOXs[i].addEstimatedXrYr(Xr, Yr)

        return BBOXs
        # return np.concatenate((preds, np.array([data[:, -1]]).T), axis=1) # concat with CAM ID 
    return []

def predict_pixel(model, MMWs): # data: np array, [[px, py, vx, vy, ax, ay], ...]
    data = np.array([[a.Px, a.Py, a.Vx, a.Vy, a.Ax, a.Ay] for a in MMWs]) # [[px, py, vx, vy, ax, ay], ...]
    if data.size != 0:
        test_dataset =  cam_mmw(data) 
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
        
        preds = []
        for x in test_loader: # tqdm(test_loader):
            x = x.to('cuda')                        
            with torch.no_grad():                   
                pred = model(x)                     
                preds.append(pred.detach().cpu())   
        preds = torch.cat(preds, dim=0).numpy() 

        for i, (Xc, Yc) in enumerate(preds):
            MMWs[i].addEstimatedXcYc(int(Xc), int(Yc)) # pixel -> int 
            # MMWs[i].Xc, MMWs[i].Yc = int(Xc), int(Yc) # pixel -> int
        
        return MMWs

        # return np.concatenate((preds, np.array([data[:, -1]]).T), axis=1) # concat with MMW ID
    return []

def predict_pixel_linear_transform(T, MMWs):
    for i, a in enumerate(MMWs):
        
        T_uv = np.matmul(T,  np.array([a.Px, a.Py, 1]))
        MMWs[i].T_Xc, MMWs[i].T_Yc= int(T_uv[0]/T_uv[2]), int(T_uv[1]/T_uv[2])
        # print(T_uv)

    return MMWs

def predict_pixel_regression(regressor, MMWs):
    for i, a in enumerate(MMWs):
        
        reg_uv = regressor.predict(np.array([[a.Px, a.Py]])) # origin 
        MMWs[i].reg_Xc, MMWs[i].reg_Yc = int(reg_uv[0][0]), int(reg_uv[0][1])
        # print(reg_uv)

    return MMWs
# if __name__ == '__main__':

    # test_dataset =  bbox2MMW_Model(np.array([[104.        , 245.        , 105.26521565, 294.57442337]]))
    # test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    # model = bbox2MMW_Model(input_dim=2).to(device)
    # model.load_state_dict(torch.load(config['save_path']))
    # preds = predict(test_loader, model, device) 
    # print(preds)