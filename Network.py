import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def get_device(device=0):
    # Selecting training device
    if torch.cuda.is_available():
        device_name = f"cuda:{device}"
        if torch.cuda.device_count() > device:
            return device_name
        else:
            print(f"No such cuda device: {device}")
    return "cpu"


class MyAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout_prob=0.5):
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
