import torch
import numpy as np


def sammons_mapping_loss(x1, x2, y1, y2):
    return torch.sum((torch.dist(x1, x2) - torch.dist(y1, y2))**2 /
                     (torch.dist(x1, x2) + 1e-5))


def mean_distance_error():
    pass


def distance_matrix(x):
    n = x.size(0)
    d = x.size(1)
    c = x.size(2)

    x_1 = x.unsqueeze(1).expand(n, n, d, c)
    x_2 = x.unsqueeze(0).expand(n, n, d, c)

    return torch.pow(x_1 - x_2, 2).sum((2, 3))


def rank_matrix(x):
    """
    Returns a rank matrix from pairwise distance matrix
    """
    m = x.argsort()
    rank = torch.zeros_like(m)

    for i in range(x.shape[0]):
        pos = torch.where(m[:, :] == i)
        rank[:, i] = pos[1]
    return rank


def trustworthiness(x, y, K=5):
    N = x.shape[0]

    d_x = distance_matrix(x)
    d_y = distance_matrix(y)

    r_x = rank_matrix(d_x)
    r_y = rank_matrix(d_y)
    tw_score = []
    for i in range(N):
        false_neighbours = set(r_x[i, 1:K + 1]).difference(r_y[i, 1:K + 1])
        score = 1 - (2 / (N * K * (2 * N - 3 * K - 1))) * sum([
            (np.where(r_x[i] == value)[0][0] - K) for value in false_neighbours
        ])
        tw_score.append(score)

    return tw_score


def continuity(x, y, K=5):
    N = x.shape[0]

    d_x = distance_matrix(x)
    d_y = distance_matrix(y)

    r_x = rank_matrix(d_x)
    r_y = rank_matrix(d_y)
    ct_score = []
    for i in range(N):
        false_neighbours = set(r_y[i, 1:K + 1]).difference(r_x[i, 1:K + 1])
        score = 1 - (2 / (N * K * (2 * N - 3 * K - 1))) * sum([
            (np.where(r_x[i] == value)[0][0] - K) for value in false_neighbours
        ])
        ct_score.append(score)

    return ct_score
