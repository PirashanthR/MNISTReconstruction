import typing

import torch


def split_left_right(
        images: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, 
                                              torch.Tensor, torch.Tensor]:
    """Given a set of half images (mnist data), we split the images on two sets:
    the ones that are left part of the images
    and the ones that are right part of the images

    This is a rule based heuristic that will certainly not work
    for other datasets
    Args:
        images (tensor): list of half images

    Returns:
        tupe(tensor, tensor, tensor, tensor): (images that are left part of 
        the images, images that are right part of the images,
        indices of left images from the raw set of images, indices of right 
        images from the raw set of images)
    """
    sum_left = images[:, :, 0].sum(axis=1)
    sum_right = images[:, :, -1].sum(axis=1)

    sum_left_large = images[:, :, :7].sum(axis=1).sum(axis=1)
    sum_right_large = images[:, :, 7:].sum(axis=1).sum(axis=1)

    sum_left = torch.where(sum_left == sum_right, sum_left_large, sum_left)
    sum_right = torch.where(
        sum_left == sum_right,
        sum_right_large,
        sum_right)
    is_right = sum_left > sum_right
    is_left = sum_right >= sum_left

    images_left_side = images[is_left]
    images_right_side = images[is_right]

    indices_left = (is_left).nonzero(as_tuple=True)
    indices_right = (is_right).nonzero(as_tuple=True)

    return (images_left_side, images_right_side,
            indices_left[0], indices_right[0])
