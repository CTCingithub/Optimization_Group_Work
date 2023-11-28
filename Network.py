'''
Author: CTC 2801320287@qq.com
Date: 2023-11-25 14:39:45
LastEditors: CTC 2801320287@qq.com
LastEditTime: 2023-11-29 00:28:06
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset,random_split
from tqdm import tqdm

def GET_DEVICE(DEVICE=0):
    # Selecting training device
    if torch.cuda.is_available():
        device_name = f"cuda:{DEVICE}"
        if torch.cuda.device_count() > DEVICE:
            return device_name
        else:
            print(f"No such cuda device: {DEVICE}")
    return "cpu"

def INIT_WEIGHTS_XAVIER(MODEL):
    # Xavier initial weights
    for name, param in MODEL.named_parameters():
        if type(MODEL) == nn.Linear:
            nn.init.xavier_uniform_(param)
            MODEL.bias.data.fill_(0.01)

def INIT_WEIGHTS_ZERO(MODEL):
    # Zero initial weights
    for param in MODEL.parameters():
        nn.init.zeros_(param)

def Split2Loaders(INPUT,OUTPUT,BATCHSIZE,RATIO=0.8,SHUFFLE=True):
    train_size = int(RATIO * INPUT.shape[0])
    test_size = INPUT.shape[0] - train_size
    train_dataset, test_dataset = random_split(TensorDataset(INPUT,OUTPUT), [train_size, test_size])
    return DataLoader(train_dataset,BATCHSIZE,SHUFFLE),DataLoader(test_dataset,BATCHSIZE,SHUFFLE)


class MyAutoencoder(nn.Module):
    # 2 Layer MLP Encoder & Decoder Attempt
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout_prob=0.1):
        super(MyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size_2),
            nn.Dropout(dropout_prob),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size_2, hidden_size_1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size_1),
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
    
def train(MODEL,NUM_EPOCHS,OPTIMIZER,TRAIN_LOADER,TEST_LOADER,LOSS_TYPE=nn.MSELoss(),DEVICE=0,GRAD_MAX=5):
    print("PyTorch Version:",torch.__version__)
    device=GET_DEVICE(DEVICE)
    print("Training on",device)
    print(
        "====================================Start training===================================="
    )
    # Transfer model to selected device
    MODEL.to(device)

    # loss recorders
    train_losses = []
    test_losses = []

    for epoch in range(NUM_EPOCHS):
        # Switch to train mode
        MODEL.train()

        # Record loss sum in 1 epoch
        LOSS_TRAIN=torch.tensor(0.0)
        LOSS_TEST=torch.tensor(0.0)

        # Gradient descent
        with tqdm(
            TRAIN_LOADER, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch"
        ) as t:
            for x, y in t:
                # Forward propagation
                x, y = x.to(device), y.to(device)
                output = MODEL(x)
                loss = LOSS_TYPE(output, y)

                # Backward propagation
                OPTIMIZER.zero_grad()
                loss.backward()

                # Gradient clipping
                clip_grad_norm_(MODEL.parameters(),GRAD_MAX)

                OPTIMIZER.step()
                t.set_postfix(loss=loss.item())
                LOSS_TRAIN+=loss.item()
        
        LOSS_TRAIN_AVERAGE=LOSS_TRAIN/len(TRAIN_LOADER)
        train_losses.append(LOSS_TRAIN_AVERAGE)

        # Model evaluation
        MODEL.eval()
        with torch.no_grad():
            for x,y in TEST_LOADER:
                x,y=x.to(DEVICE),y.to(DEVICE)
                output=MODEL(x)
                loss=LOSS_TYPE(output,y)
                LOSS_TEST+=loss.item()
        
        LOSS_TEST_AVERAGE=LOSS_TEST/len(TEST_LOADER)
        test_losses.append(LOSS_TEST_AVERAGE)

    print(
        "====================================Finish training====================================\n"
    )

    return train_losses,test_losses
