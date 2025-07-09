import torch
import numpy as np


def get_cls_bin_edges(n_classes):
    """
    Get the bin edges for the classification mask.
    Args:
    n_classes (int): Number of classes in the classification mask.
    Returns:
    bin_edges (list): List of bin edges for the classification mask.
    All values are predefined and fixed during training (do not change them).
    """
    if n_classes == 11:
        bin_edges = [
            np.log1p(1),
            np.log1p(50),
            np.log1p(150),
            np.log1p(300),
            np.log1p(450),
            np.log1p(750),
            np.log1p(1000),
            np.log1p(1500),
            np.log1p(2000),
            np.log1p(2500),
            np.log1p(3500),
        ]

    elif n_classes == 7:
        bin_edges = [
            np.log1p(1),
            np.log1p(100),
            np.log1p(300),
            np.log1p(500),
            np.log1p(1000),
            np.log1p(2000),
            np.log1p(3500),
        ]

    else:
        raise ValueError("Invalid number of classes")

    return bin_edges


def normalize_df(df, normalizer):
    return -torch.log(df / normalizer + 1e-6)


def denormalize_df(df_norm, normalizer):
    return torch.exp(-df_norm) * normalizer


def reverse_log_transform(y):
    return torch.expm1(y)  # expm1(x) = exp(x) - 1


def log_transform_mask(y):
    return np.log1p(y)  # log1p(x) = log(1 + x)


def reverse_log_transform_mask(y):
    return np.expm1(y)  # expm1(x) = exp(x) - 1


def min_max_scale(y, min_val, max_val):
    return (y - min_val) / (max_val - min_val)


def reverse_min_max_scale(y, min_val, max_val):
    return y * (max_val - min_val) + min_val
