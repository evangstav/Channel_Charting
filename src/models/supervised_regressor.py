import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score

from tqdm import tqdm


class Regressor(nn.Module):
    def __init__(self, channels):
        super(Regressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels,
                               out_channels=128,
                               kernel_size=8,
                               stride=2)
        self.conv2 = nn.Conv1d(in_channels=128,
                               out_channels=64,
                               kernel_size=4,
                               stride=2)
        self.conv3 = nn.Conv1d(in_channels=64,
                               out_channels=32,
                               kernel_size=2,
                               stride=2)
        self.lin1 = nn.Linear(160, 64)
        self.lin2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        x = F.normalize(F.relu(self.conv1(x)))
        x = F.normalize(F.relu(self.conv2(x)))
        x = F.normalize(F.relu(self.conv3(x)))
        #x = F.avg_pool1d(x, kernel_size=3)
        x = torch.flatten(x, 1)
        x = F.dropout(F.selu(self.lin1(x)), 0.2)
        x = F.dropout(F.selu(self.lin2(x)), 0.2)

        x = self.out(x)
        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    loss = 0
    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        batch_loss = criterion(y, y_hat)

        batch_loss.backward()

        optimizer.step()

        loss += batch_loss.item()
    loss /= len(train_loader.dataset)

    return loss


def test(model, test_loader, criterion, device):
    model.eval()
    val_loss = 0
    accuracy = 0
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        val_loss += criterion(y_hat, y).item()
    return val_loss / len(test_loader.dataset)
