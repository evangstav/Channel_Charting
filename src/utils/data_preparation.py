import torch
from torch.utils.data import Dataset
import numpy as np
import random


class SiameseDataset(Dataset):
    def __init__(self, data, scaler=None):
        """
        data: Numpy array
        scaler: (None or (scaler_real, scaler_imag)
        """
        self.raw_data = data
        #real, imag
        self.real = np.real(data).astype(np.float32)
        self.imag = np.imag(data).astype(np.float32)

        # Normalize
        if not scaler:
            self.scaler_real = SimpleStandardScaler().fit(self.real)
            self.scaler_imag = SimpleStandardScaler().fit(self.imag)
        else:
            self.scaler_real, self.scaler_imag = scaler

        self.real = self.scaler_real.transform(self.real)
        self.imag = self.scaler_imag.transform(self.imag)

        self.full_data = torch.from_numpy(
            np.concatenate([self.real, self.imag], axis=1))

    def __len__(self):
        """
        returns number of samples in dataset
        """
        return self.full_data.shape[0]

    def nb_channels(self):
        """
        returns number of channel on dataset
        """
        return self.full_data.shape[1]

    def nb_samples(self):
        """
        returns number of samples for each timeseries
        """
        return self.full_data.shape[2]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = list(idx.tolist())

        sample_1 = self.full_data[idx]
        # take a random sample
        if type(idx) is int:
            random_idx = np.random.randint(0, len(self))
        else:
            random_idx = torch.randperm(len(sample_1))

        sample_2 = self.full_data[random_idx]

        return sample_1, sample_2


class SimpleStandardScaler():
    """
    Simple scaler that normalises data
    input: numpy array
    output: numpy array
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std
