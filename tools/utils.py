import numpy as np


def load_datasets():
    """
    Load all datasets
    """
    # load frequency data
    taps = np.load("../Data/RayTracingData/Remcom_4x4_IR_100taps.npy")
    # load Phi and Theta
    phi = np.load('../Data/RayTracingData/Remcom_4x4_AoA_phi.npy')
    theta = np.load('../Data/RayTracingData/Remcom_4x4_AoA_theta.npy')

    # load receiver positions
    rx_positions = np.load("../Data/RayTracingData/Remcom_4x4_rxpos.npy")
    # load transmitter positions
    tx_positions = np.load("../Data/RayTracingData/Remcom_4x4_txpos.npy")

    return taps, phi, theta, rx_positions, tx_positions


def standarize(x):
    return (np.array(x) - np.mean(x)) / np.std(x)


def drop_top_right(data, rx_positions):
    idxx = rx_positions[:, 0] > 300
    idxy = rx_positions[:, 1] > 150
    idx = np.logical_and(idxx, idxy)
    good_idcs = ~idx
    return data[good_idcs]


def fillna(x, value=0):
    x[np.where(np.isnan(x))] = value
    return x


def zero_padding_as(x, target):
    width = (target.shape[2] - x.shape[2]) // 2
    x = np.pad(x, (width, width))
    return x


def random_sample_and_remove(X, y, sample_size):
    """A function that takes a random subset of samples out of a numpy array
    inputs: (X::np.array)
            (y::np.array)
            (sample_size: (integer))
    outputs: subset_X::np.array
             subset_y::np.array
             (original_X - subset_X)::np.array
             (original_y - subset_y)::np.array
    """
    indices = np.random.choice(data.shape[0], sample_size, replace=False)
    return (X[indices], X[~indices], y[indices], y[~indices])


def take_norm(x):
    return np.absolute(x)
