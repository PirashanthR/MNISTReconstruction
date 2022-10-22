import typing
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def create_pairs_affinity_matrix(affinity_matrix: np.array,
                                 indices_left: torch.Tensor,
                                 indices_right: torch.Tensor)\
                                    -> typing.List[typing.Tuple[int, int]]:
    """Given an affinity matrix, generate the final pairs that associates two
    half images for reconstructing the original mnist full images using hungarian algorithm 

    Args:
        affinity_matrix (array): affinity matrix (should be of length indices 
                                left x indices right)
        indices_left (tensor): indices of left images from 
                                the raw set of images
        indices_right (tensor): indices of right images from 
                                the raw set of images

    Returns:
        list(tuple): list of pairs associated to reconstruct mnist data
    """

    row_idx, col_idx = linear_sum_assignment(affinity_matrix, maximize=True)

    list_pairs = []
    for (idx_left, idx_row) in zip(row_idx, col_idx):
            list_pairs.append((indices_left[idx_left].item(),
                               indices_right[idx_row].item()))
    return list_pairs
