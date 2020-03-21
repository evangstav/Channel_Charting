import wandb

import argparse
import numpy as np
import scipy.fft as fft
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from models import siamese
from utils import data_preparation, metrics

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--input_channels', type=int, default=32)
parser.add_argument('--input_seq_len', type=int, default=100)
parser.add_argument('--channels_1', type=int, default=128)
parser.add_argument('--channels_2', type=int, default=64)
parser.add_argument('--channels_3', type=int, default=32)
parser.add_argument('--stride_1', type=int, default=1)
parser.add_argument('--stride_2', type=int, default=1)
parser.add_argument('--stride_3', type=int, default=1)
parser.add_argument('--kernel_1', type=int, default=32)
parser.add_argument('--kernel_2', type=int, default=16)
parser.add_argument('--kernel_3', type=int, default=8)
parser.add_argument('--lin_features_1', type=int, default=128)
parser.add_argument('--lin_features_2', type=int, default=64)
parser.add_argument('--lin_features_3', type=int, default=32)
parser.add_argument('--mapping_dim', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--used_pct', type=float, default=0.9)

args = parser.parse_args()
wandb.init(project='ChannelCharting', config=args)

input_channels = args.input_channels
input_seq_len = args.input_seq_len
epochs = args.epochs
batch_size = args.batch_size
used_pct = args.used_pct

net_config = (args.channels_1, args.channels_2, args.channels_3, args.stride_1,
              args.stride_2, args.stride_3, args.kernel_1, args.kernel_2,
              args.kernel_3, args.lin_features_1, args.lin_features_2,
              args.lin_features_3, args.mapping_dim, args.dropout)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

model = siamese.SiameseNN(input_channels, input_seq_len, net_config).to(device)

# read data
data = np.load("../Data/RayTracingData/Remcom_4x4_IR_100taps.npy")
# read positions
positions = np.load("../Data/RayTracingData/Remcom_4x4_rxpos.npy")

# take a sample of the dataset
idces = np.random.randint(0, data.shape[0], int(used_pct * data.shape[0]))
data_undersampled = data[idces]
print(data_undersampled.shape)
#Fourier transform and smoothen by undersampling
data_undersampled = fft.fft(data_undersampled)
data_undersampled = data_undersampled[:, :, ::2]
positions_undersampled = positions[idces]
print(data_undersampled.shape, positions_undersampled.shape)

train_ds, test_ds, train_y, test_y = train_test_split(data_undersampled,
                                                      positions_undersampled,
                                                      test_size=0.2)

train_dataset = data_preparation.SiameseDataset(train_ds)
scaler = train_dataset.scaler_real, train_dataset.scaler_imag
test_dataset = data_preparation.SiameseDataset(test_ds, scaler_X=scaler)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
criterion = metrics.sammons_mapping_loss
optimizer = optim.Adam(model.parameters())
scheduler = StepLR(optimizer, step_size=1)

wandb.watch(model)
for e in range(1, epochs + 1):
    train_loss = siamese.train(model, device, train_loader, optimizer,
                               criterion)
    validation_loss, ct, tw = siamese.test(model, device, test_loader,
                                           criterion)
    scheduler.step()

    plt.scatter(
        model(test_dataset[:500][0]).detach()[:, 0],
        model(test_dataset[:500][0]).detach()[:, 1])

    wandb.log({
        "Training Loss": train_loss,
        "Validation Loss": validation_loss,
        "Continuity": ct,
        "Trustworthiness": tw,
        "Projected Space": plt
    })
