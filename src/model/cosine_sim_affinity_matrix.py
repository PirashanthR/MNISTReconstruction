import numpy as np
import scipy.spatial as sp
import torch
from src.utils import split_left_right


def affinity_matrix_pair(images_left: np.array, images_right: np.array) -> np.array:
    """Given a set of images that are supposed to be left part of images and a
    set of images that are supposed to be right part of images.
    The function computes the affinity matrix that represents
    the probability that image i is linked to image j

    Args:
        images_left (array): image array
        images_right (array): image array

    Returns:
        affinity matrix: array
    """
    matrix_left = images_left[:, :, -1]
    matrix_right = images_right[:, :, 0]

    af = sp.distance.cdist(matrix_left, matrix_right, 'cosine')
    af[np.isnan(af)] = 1
    return 1 - af


def generate_affinity_matrix(images: torch.Tensor):
    images_left_side, images_right_side, indices_left, indices_right = \
        split_left_right(images)
    af = affinity_matrix_pair(images_left_side, images_right_side)
    return af, indices_left, indices_right
