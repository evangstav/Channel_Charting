import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.fft as fft
import seaborn as sns
sns.set()

from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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


def get_k_best(predictions, K):
    """Function that return the indices of most K most confident predictions"""
    K_preds = sorted(list(enumerate(predictions.max(dim=1).values.detach())),
                     key=lambda x: x[1],
                     reverse=True)[:K]
    return [x[0] for x in K_preds]


def get_k_worst(predictions, K):
    """Function that return the indices of most K most confident predictions"""
    K_preds = sorted(list(enumerate(predictions.max(dim=1).values.detach())),
                     key=lambda x: x[1],
                     reverse=False)[:K]
    return [x[0] for x in K_preds]


# configuration
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
batch_size = 64
train_size = 5000
n_clusters = 8
K_best = 100

# load datasets
(raw_taps, raw_phi, raw_theta, raw_rx_positions,
 raw_tx_positions) = utils.load_datasets()

#fourier transform and undersample taps
raw_freq_taps = fft.fft(raw_taps, workers=-1)[:, :, ::2]
taps = preprocessing(raw_freq_taps, raw_freq_taps, raw_rx_positions)
taps = np.hstack([np.real(taps), np.imag(taps)])
phi = preprocessing(raw_phi, taps, raw_rx_positions)
theta = preprocessing(raw_theta, taps, raw_rx_positions)

y = preprocessing(raw_rx_positions, taps, raw_rx_positions,
                  padding=False)[:, :2]
X = np.hstack([taps, phi[:-10], theta[:-10]])

#assign labels to certain areas of the map using kmeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters=n_clusters)
km = km.fit(y)
labels = km.predict(y)

#train test split
# keep 25% of the dataset for testing
train_X, test_X, train_y, test_y, train_labels, test_labels = train_test_split(
    X, y, labels)
test_DS = SupervisedDataset(test_X, test_labels)
test_loader = DataLoader(test_DS, batch_size=batch_size)

# use 2500 ~ 10% of the original dataset as labeled dataset
(X_sampled, X_remaining, y_sampled, y_remaining, labels_sampled,
 labels_remaining) = train_test_split(train_X,
                                      train_y,
                                      train_labels,
                                      train_size=train_size)
its = 0
while len(X_remaining) > train_size:

    train_DS = SupervisedDataset(X_sampled, labels_sampled)
    train_loader = DataLoader(train_DS, batch_size=batch_size, shuffle=True)

    if its == 0:
        model = supervised_classifier.Classifier(channels=train_DS.channels(),
                                                 nb_labels=n_clusters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
    best_val_loss = 9999
    count = 0

    for epoch in range(50):
        loss, acc = supervised_classifier.train(model, train_loader, optimizer,
                                                criterion, device)
        val_loss, val_acc = supervised_classifier.test(model, test_loader,
                                                       criterion, device)
        print(
            f"Epoch: {epoch}, Training Loss: {loss}, Training Accuracy: {acc} ,Validation Loss: {val_loss}, Validation Accuracy: {val_acc}"
        )

        if best_val_loss < val_loss:
            count += 1
        else:
            best_val_loss = val_loss
        if count > 5:
            break

    # add confident samples to training
    # generate predictions for the remaining dataset
    yhats = model(torch.Tensor(X_remaining))
    #get indices for the most confident predictions
    idces = get_k_best(yhats, K_best)
    # take the samples corresponding to those predictions and add them to the dataset
    (confident_X, confident_labels, confident_y) = (
        X_remaining[idces],
        # get the labels of the most confident predictions
        yhats[idces].argmax(dim=1).detach().numpy(),
        y_remaining[idces])
    # concatenate the old and the new samples
    X_sampled = np.concatenate([X_sampled, confident_X])
    y_sampled = np.concatenate([y_sampled, confident_y])
    labels_sampled = np.concatenate([labels_sampled, confident_labels])

    mask_array = np.zeros(len(X_remaining), dtype=bool)
    mask_array[idces] = True
    # remove the added samples from the remaining
    X_remaining, labels_remaining, y_remaining = (
        X_remaining[~mask_array], labels_remaining[~mask_array],
        y_remaining[~mask_array])

    print("Remaining Dataset", len(X_remaining))
    its += 1

#test
yhats = model(test_DS[:][0]).detach().argmax(dim=1)
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.scatterplot(test_y[:, 0],
                test_y[:, 1],
                hue=yhats,
                legend="full",
                palette='gnuplot')
plt.subplot(1, 2, 2)
sns.scatterplot(test_y[:, 0],
                test_y[:, 1],
                hue=test_labels,
                legend="full",
                palette='gnuplot')
plt.show()
print(classification_report(yhats, test_labels))
sns.clustermap(confusion_matrix(yhats, test_labels), cmap='Blues')
plt.show()
