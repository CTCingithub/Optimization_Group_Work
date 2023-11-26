'''
Author: CTC 2801320287@qq.com
Date: 2023-11-25 14:39:45
LastEditors: CTC 2801320287@qq.com
LastEditTime: 2023-11-26 17:26:02
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def GET_DEVICE(device=0):
    # Selecting training device
    if torch.cuda.is_available():
        device_name = f"cuda:{device}"
        if torch.cuda.device_count() > device:
            return device_name
        else:
            print(f"No such cuda device: {device}")
    return "cpu"


class MyAutoencoder(nn.Module):
    # 3 Layer MLP Encoder & Decoder Attempt
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout_prob=0.5):
        super(MyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size_2),
            nn.Dropout(dropout_prob),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size_2, hidden_size_1),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size_1, input_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, ENCODER, DECODER):
        super(MyAutoencoder, self).__init__()
        self.encoder = ENCODER
        self.decoder = DECODER
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x