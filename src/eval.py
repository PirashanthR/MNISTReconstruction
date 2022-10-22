import typing

import numpy as np
import torch
from tqdm import tqdm


def custom_matching_score(
        list_pairs: typing.List[typing.Tuple[int, int]],
        y_true: torch.Tensor) -> float:
    """Matching score corresponding to what is described in the task 
    description"""
    assert(len(list_pairs)*2 == len(y_true))

    nb_ok = 0
    for pair in list_pairs:
        if y_true[pair[0]] == y_true[pair[1]]:
            nb_ok += 1
    return nb_ok/len(list_pairs)


def generate_y_pred(
        list_pairs: typing.List[typing.Tuple[int, int]]) -> np.array:
    """generate prediction based on list pairs"""
    y_pred = np.zeros(2*len(list_pairs))

    for i, p in enumerate(tqdm(list_pairs)):
        y_pred[p[0]] = i
        y_pred[p[1]] = i
    return y_pred