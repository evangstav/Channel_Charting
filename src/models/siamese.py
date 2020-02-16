import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../utils')
from utils import metrics


def conv1d_output_size(size, kernel_size=1, stride=1, pad=0, dilation=1):
    return (size + (2 * pad) - (dilation *
                                (kernel_size - 1)) - 1) // stride + 1


class SiameseNN(nn.Module):
    def __init__(self, input_channels, input_seq_len, config):
        super(SiameseNN, self).__init__()

        (channels_1, channels_2, channels_3, kernel_1, kernel_2, kernel_3,
         stride_1, stride_2, stride_3, lin_features_1, lin_features_2,
         lin_features_3, mapping_dim, dropout) = config
        self.conv1 = nn.Conv1d(in_channels=input_channels,
                               out_channels=channels_1,
                               kernel_size=kernel_1,
                               stride=stride_1)
        self.bn1 = nn.BatchNorm1d(channels_1)
        self.conv2 = nn.Conv1d(in_channels=channels_1,
                               out_channels=channels_2,
                               kernel_size=kernel_2,
                               stride=stride_2)
        self.bn2 = nn.BatchNorm1d(channels_2)
        self.conv3 = nn.Conv1d(in_channels=channels_2,
                               out_channels=channels_3,
                               kernel_size=kernel_3,
                               stride=stride_3)
        self.bn3 = nn.BatchNorm1d(channels_3)

        f = conv1d_output_size
        self.features = f(f(f(input_seq_len,
                              kernel_size=kernel_1,
                              stride=stride_1),
                            kernel_size=kernel_2,
                            stride=stride_2),
                          kernel_size=kernel_3,
                          stride=stride_3)

        self.lin1 = nn.Linear(in_features=channels_3 * self.features,
                              out_features=lin_features_1)
        self.lin2 = nn.Linear(in_features=lin_features_1,
                              out_features=lin_features_2)
        self.lin3 = nn.Linear(in_features=lin_features_2,
                              out_features=lin_features_3)
        self.lin4 = nn.Linear(in_features=lin_features_3,
                              out_features=mapping_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.lin1(x)))
        x = self.dropout(F.relu(self.lin2(x)))
        x = self.dropout(F.relu(self.lin3(x)))

        out = self.lin4(x)

        return out


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for x1, x2 in train_loader:
        x1, x2 = x1.to(device), x2.to(device)
        optimizer.zero_grad()
        y1, y2 = model(x1), model(x2)

        loss = criterion(x1, x2, y1, y2)

        loss.backward()

        optimizer.step()

        epoch_loss += loss
    return epoch_loss.item()


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    ct = 0
    tw = 0
    for x1, x2 in test_loader:
        x1, x2 = x1.to(device), x2.to(device)
        y1, y2 = model(x1), model(x2)
        test_loss += criterion(x1, x2, y1, y2)
        y1 = y1.unsqueeze(2).detach()
        ct += np.mean(metrics.continuity(x1, y1, range(1, 5)))
        tw += np.mean(metrics.trustworthiness(x1, y1, range(1, 5)))

    return (test_loss / len(test_loader.dataset),
            ct / len(test_loader.dataset), tw / len(test_loader.dataset))
