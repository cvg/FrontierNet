import numpy as np
from .helpers import denormalize_df, get_cls_bin_edges


def df2region(df, normalizer=10, threshold=0.5):
    """
    Convert a distance field to a binary frontier region mask.

    Args:
    df (torch.Tensor): The distance field as a 2D tensor.
    normalizer (float): Normalization factor for the distance field.
    threshold (float): Threshold value to detect lines.

    Returns:
    region_mask (numpy array): A binary mask where lines are frontiers.
    """
    df = denormalize_df(df, normalizer=normalizer).squeeze().cpu().numpy()
    region_mask = np.zeros_like(df)
    region_mask[df < threshold] = 1

    return region_mask


def cls2gainmap(label_mask, bin_edges):
    """
    Convert a classification mask to a gain map.
    Args:
    label_mask (numpy array): The classification mask as a 2D tensor.
    bin_edges (list): The edges of the bins for classification.
    Returns:
    gain_map (numpy array): The gain map corresponding to the classification prediction.
    """
    # Calculate the average of the bin edges for each bin
    bin_values = [
        (bin_edges[i - 1] + bin_edges[i]) / 2 for i in range(1, len(bin_edges))
    ]
    gain_map = np.zeros_like(label_mask, dtype=np.float32)

    # Map each label in the label_mask back to the corresponding bin value
    for i in range(1, len(bin_edges)):
        gain_map[label_mask == i] = bin_values[i - 1]

    # Apply the exponential transform to reverse the log1p operation
    gain_map = np.expm1(gain_map)

    return gain_map


def prediction2frontiermap(df, cls_mask, n_classes=11, df_normalizer=10, threshold=0.5):
    """
    Convert the output of the Unet model to a 2D frontier map.
    This corresponds to equation (6) and (7) in the paper.
    i.e., from D, Y to F and G
    Args:
    df (torch.Tensor): The distance field as a 2D tensor. (D in paper)
    cls_mask (torch.Tensor): The classification mask as a 2D tensor. (Y in paper)
    n_classes (int): Number of classes in the classification mask. (K in paper)
    df_normalizer (float): Normalization factor for the distance field.
    threshold (float): Threshold value to detect lines.
    Returns:
    region (numpy array): A binary mask where lines are frontiers. (F in paper)
    gain (numpy array): The weight map for the frontier. (G in paper)
    """

    region = df2region(df, df_normalizer, threshold)
    cls_mask = cls_mask.argmax(dim=1, keepdim=False).squeeze().cpu().numpy()
    gain_map = cls2gainmap(cls_mask, get_cls_bin_edges(n_classes=n_classes))
    gain = np.zeros_like(gain_map)
    gain[region > 0] = gain_map[region > 0]

    return region, gain
