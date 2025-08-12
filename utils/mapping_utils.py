import torch
import torch.nn.functional as F
import math
import numpy as np


def rasterize_voxel_depth(
    vx_centers,
    radius,
    K,
    T_cams_world,
    image_size,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Fast batched voxel rasterization using scatter_reduce and physically-based radius expansion.

    Args:
        vx_centers (np.ndarray): (N, 3) world-space voxel centers
        radius (float): 3D voxel radius (in meters)
        K (np.ndarray): (3, 3) camera intrinsics
        T_cams_world (np.ndarray): (B, 4, 4) camera extrinsics (camera_T_world)
        image_size (tuple): (H, W)
        device (str): 'cuda' or 'cpu'

    Returns:
        depth_imgs: (B, H, W) numpy array of depth images
    """
    B = T_cams_world.shape[0]
    H, W = image_size

    # Early return if no voxels
    if vx_centers is None or len(vx_centers) == 0:
        return np.full((B, H, W), 100, dtype=np.float32)

    N = vx_centers.shape[0]

    # Prepare tensors
    vx_centers = torch.tensor(vx_centers, dtype=torch.float32, device=device)  # (N, 3)
    K = torch.tensor(K, dtype=torch.float32, device=device)
    T = torch.tensor(T_cams_world, dtype=torch.float32, device=device)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Homogeneous coordinates
    ones = torch.ones((N, 1), device=device)
    vx_centers_h = (
        torch.cat([vx_centers, ones], dim=1).unsqueeze(0).expand(B, N, 4)
    )  # (B, N, 4)

    # Transform world to camera frame
    xyz_cam = torch.bmm(vx_centers_h, T.transpose(1, 2))[:, :, :3]  # (B, N, 3)
    x, y, z = xyz_cam.unbind(-1)  # (B, N)

    # Keep only points in front of the camera
    valid = z > 0.1
    cam_idx, pt_idx = valid.nonzero(as_tuple=True)
    x, y, z = x[valid], y[valid], z[valid]

    # Project to 2D
    u = (fx * x / z + cx).round().long()
    v = (fy * y / z + cy).round().long()

    # Filter projections within image bounds
    valid_u = (u >= 0) & (u < W)
    valid_v = (v >= 0) & (v < H)
    valid_proj = valid_u & valid_v

    cam_idx = cam_idx[valid_proj]
    u = u[valid_proj]
    v = v[valid_proj]
    z = z[valid_proj]

    # Allocate full flattened image buffer for all B batches
    depth_flat = torch.full((B * H * W,), float("inf"), device=device)

    # Offset each pixel index by batch ID
    offsets = cam_idx * (H * W)
    pixel_idx_global = (v * W + u) + offsets

    # Perform a single scatter_reduce over all batches
    depth_flat.scatter_reduce_(
        dim=0, index=pixel_idx_global, src=z, reduce="amin", include_self=True
    )

    # Reshape back to (B, H, W)
    depth_imgs = depth_flat.view(B, H, W)
    depth_imgs[depth_imgs == float("inf")] = 0.0

    # Compute Physical Radius in Pixels
    if z.numel() == 0:
        # nothing was projected -> return current image as-is
        return (
            depth_imgs.cpu().numpy()
            if device.startswith("cuda")
            else depth_imgs.numpy()
        )

    projected_r_pix = fx * radius / z.clamp(min=1e-4)
    max_r_pix = projected_r_pix.max().item()
    kernel_size = int(2 * math.ceil(max_r_pix) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius_px = kernel_size // 2

    # Radius Expansion with Min-Pooling
    assert radius_px > 0
    depth = depth_imgs.clone()
    depth[depth == 0.0] = float("inf")

    depth_inv = -depth.unsqueeze(1)  # (B, 1, H, W)
    depth_expanded_inv = F.max_pool2d(
        depth_inv, kernel_size=kernel_size, stride=1, padding=radius_px
    )
    depth_expanded = -depth_expanded_inv.squeeze(1)
    depth_expanded[depth_expanded == float("inf")] = 0.0

    return (
        depth_expanded.cpu().numpy()
        if device.startswith("cuda")
        else depth_expanded.numpy()
    )


def render_voxel_depth(occ_pts, camera_extrinsics, render_params):
    """
    Render depth images from the occupied voxels
    camera_extrinsics: (B, 4, 4) numpy array of camera extrinsics (world to camera)
    """
    depth_imgs = rasterize_voxel_depth(
        occ_pts,
        radius=render_params["radius"],
        K=render_params["K"],
        T_cams_world=camera_extrinsics,
        image_size=(render_params["H"], render_params["W"]),
    )
    return depth_imgs


def select_visible_points(points, K, T, img_size, max_depth=np.inf, min_depth_map=None):
    """
    Project 3D points to 2D image coordinates using camera intrinsics and extrinsics,
    returning only those that are visible and optionally closer than a depth map.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 3)), np.array([])

    points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)
    valid_points = np.ones(points.shape[0], dtype=bool)

    # Transform to camera coordinates
    points_cam = (T @ points_h.T).T[:, :3]  # (N, 3)
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    # Filter by depth range
    valid_points &= (z > 0) & (z <= max_depth)

    # Project to image plane
    u = np.round(K[0, 0] * x / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * y / z + K[1, 2]).astype(int)

    # Filter out-of-bounds pixels
    in_bounds = (u >= 0) & (u < img_size[1]) & (v >= 0) & (v < img_size[0])
    valid_points &= in_bounds

    # Filter based on depth map
    if min_depth_map is not None:
        depth_map_values = np.full(points.shape[0], np.nan)
        depth_map_values[in_bounds] = min_depth_map[v[in_bounds], u[in_bounds]]
        visible_mask = np.isnan(depth_map_values) | (z <= depth_map_values)
        valid_points &= visible_mask

    valid_indices = np.where(valid_points)[0]
    return points[valid_indices], valid_indices


def compute_visible_voxels(
    K, T, img_size, voxel_size=0.1, max_depth=np.inf, min_depth_map=None
):
    """
    Calculate the maximum number of voxels that can be visible in an image based on camera parameters and optional depth map.
    """

    # create voxel grid in camera coordinates
    z = np.arange(voxel_size, max_depth, voxel_size)
    x_range = int(0.8 * (max_depth * (img_size[1] / 2)) / K[0, 0])
    y_range = int(0.8 * (max_depth * (img_size[0] / 2)) / K[1, 1])
    x = np.arange(-x_range, x_range + voxel_size, voxel_size)
    y = np.arange(-y_range, y_range + voxel_size, voxel_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    voxel_centers = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  # (N, 3)

    # transform voxel centers to world coordinates
    voxel_centers_h = np.hstack((voxel_centers, np.ones((voxel_centers.shape[0], 1))))
    voxel_centers_world = (np.linalg.inv(T) @ voxel_centers_h.T).T[:, :3]

    # handle strict zero depth
    if min_depth_map is not None:
        min_depth_map = np.array(min_depth_map, dtype=np.float32)
        min_depth_map[abs(min_depth_map) < 1e-3] = max_depth

    # project to image and get visible ones
    valid_voxels, valid_indices = select_visible_points(
        voxel_centers_world,
        K=K,
        T=T,
        img_size=img_size,
        max_depth=max_depth,
        min_depth_map=min_depth_map,
    )

    return voxel_centers_world[valid_indices], valid_voxels.shape[0]
