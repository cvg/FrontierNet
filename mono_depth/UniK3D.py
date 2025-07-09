from unik3d.models import UniK3D
from unik3d.utils.camera import (Pinhole, OPENCV, Fisheye624, MEI, Spherical)
import numpy as np
import torch


def metric_depth_from_rgb(rgb_input, intrinsic_mat):
    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl")  # vitl for ViT-L backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    rgb = (
        torch.from_numpy(np.array(rgb_input).astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    if intrinsic_mat is None:
        predictions = model.infer(rgb)
    else:
        cam_params = [
            intrinsic_mat[0, 0],  # fx
            intrinsic_mat[1, 1],  # fy
            intrinsic_mat[0, 2],  # cx
            intrinsic_mat[1, 2],  # cy
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        params = torch.tensor(cam_params).to(device)
        name = "OPENCV"
        camera = eval(name)(params=params)
        predictions = model.infer(rgb, camera)

    depth = predictions["depth"]
    return depth.squeeze().cpu().numpy()
