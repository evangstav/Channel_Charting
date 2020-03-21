import wandb

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.fft as fft
import seaborn as sns

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sns.set()

from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tqdm.notebook import tqdm

import sys
sys.path.append("/home/evangelos/workspace/Channel_Charting/")
from tools import utils
from src.models import supervised_classifier, supervised_regressor
from src.utils.data_preparation import SupervisedDataset


def preprocessing(data, first_data, rx_positions, padding=True):
    data = utils.drop_top_right(data, rx_positions)
    data = utils.standarize(data)
    data = utils.fillna(data)
    if padding:
        data = utils.zero_padding_as(data, first_data)
    #data = utils.take_norm(data)

    return data


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--clusters', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--used_pct', type=float, default=1)

args = parser.parse_args()
wandb.init(project='ChannelCharting', config=args)

epochs = args.epochs
batch_size = args.batch_size
used_pct = args.used_pct

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# load datasets
(raw_taps, raw_phi, raw_theta, raw_rx_positions,
 raw_tx_positions) = utils.load_datasets()
#fourier transform and undersample taps
raw_freq_taps = fft.fft(raw_taps, workers=-1)[:, :, ::2]

# Preprocess datasets
taps = preprocessing(raw_freq_taps, raw_freq_taps, raw_rx_positions)
taps = np.hstack([np.real(taps), np.imag(taps)])
phi = preprocessing(raw_phi, taps, raw_rx_positions)
theta = preprocessing(raw_theta, taps, raw_rx_positions)
X = np.hstack([taps, phi[:-10], theta[:-10]])
y = preprocessing(raw_rx_positions, taps, raw_rx_positions,
                  padding=False)[:, :2]

#assign labels to certain areas of the map using kmeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters=args.clusters)
km = km.fit(y)
labels = km.predict(y)

#train test split
train_X, test_X, train_y, test_y, train_labels, test_labels = train_test_split(
    X, y, labels)
train_DS = SupervisedDataset(train_X, train_labels)
test_DS = SupervisedDataset(test_X, test_labels)

model = supervised_classifier.Classifier(channels=train_DS.channels(),
                                         nb_labels=args.clusters)
criterion = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(train_DS, batch_size=32)
test_loader = DataLoader(test_DS)

optimizer = optim.Adam(model.parameters())
scheduler = StepLR(optimizer, step_size=1)

wandb.watch(model)
best_val_loss = 9999
count = 0
best_accuracy = 0
for e in (range(100)):
    loss = supervised_classifier.train(model, train_loader, optimizer,
                                       criterion, device)
    val_loss, val_acc = supervised_classifier.test(model, test_loader,
                                                   criterion, device)
    # print(f"Epoch {epoch+1}: Training Loss {loss}, Validation Loss {val_loss}, Validation Accuracy {val_acc}")
    wandb.log({
        "Training Loss": loss,
        "Validation Loss": val_loss,
        "Validation Accuracy": val_acc
    })

    if best_val_loss < val_loss:
        count += 1
    else:
        best_val_loss = val_loss

    if val_acc > best_accuracy:
        wandb.run.summary["best_accuracy"] = val_acc

        yhats = model(test_DS[:][0]).detach()
        predictions = yhats.argmax(dim=1)
        report = classification_report(predictions,
                                       test_labels,
                                       output_dict=True)

        wandb.log(report)

        heatmap = sns.heatmap(confusion_matrix(predictions, test_labels))
        wandb.log({"Confusion Matrix": heatmap})

        map = sns.scatterplot(test_y[:, 0], test_y[:, 1], hue=predictions)
        wandb.log({"Map": map})

    if count > 5:
        break
