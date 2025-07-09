import numpy as np
import cv2
from typing import Tuple
from scipy.spatial.transform import Rotation as R


# Rotation/Transformation
EPS = 1e-8  # numerical tolerance


def rot2quat(rot):
    quat = R.from_matrix(rot).as_quat()  # xyzw
    return quat


def quat2rot(quat):
    rot = R.from_quat(quat).as_matrix()  # 3x3
    return rot


def _safe_normalize(v, eps=EPS):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector encountered.")
    return v / n


def _rot_between(u, v, eps=EPS):
    """
    Small-angle-safe rotation that takes `u` onto `v`
    (both assumed unit).
    """
    dot = np.dot(u, v)
    if dot > 1 - eps:  # already aligned
        return R.identity()
    if dot < -1 + eps:  # opposite; pick any orthogonal axis
        # choose an axis least parallel to u for numerical stability
        helper = (
            np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        )
        axis = _safe_normalize(np.cross(u, helper))
        return R.from_rotvec(axis * np.pi)

    axis = _safe_normalize(np.cross(u, v))
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return R.from_rotvec(axis * angle)


def compute_alignment_transforms(
    origins,
    align_vec,
    align_axis,
    appr_vec,
    appr_axis,
    refine=True,
    eps=EPS,
    reortho=True,
):
    """
    A function to compute a list of 4x4 homogeneous transforms
    that align a set of origins with a specified direction and approach vector.
    The transforms are computed such that its align_axis aligns with the align_vec,
    and its approach axis try to align with the appr_vec as much as possible.
    Parameters:
    ----------
    origins : list[np.ndarray]
        List of 3D points (origins) where the transforms will be applied.
    align_vec : np.ndarray
        The target direction vector to align with.
    align_axis : np.ndarray
        The axis along which the align_vec should be aligned.
    appr_vec : np.ndarray
        The target approach vector to align with as much as possible.
    appr_axis : np.ndarray
        The axis along which the appr_vec should be aligned as much as possible.
    refine : bool, optional
        If True, applies an additional refinement step to align the approach vector.
        Default is True.
    eps : float, optional
        Numerical tolerance for vector normalization and rotation calculations.
        Default is 1e-8.
    reortho : bool, optional
        If True, re-orthonormalizes the resulting rotation matrix to ensure it is orthonormal.
        This can help eliminate numerical drift in the rotation matrix.
        Default is True.

    Returns
    -------
    list[np.ndarray] one 4x4 homogeneous transform per origin.
    """

    # align_axis and appr_axis should be orthonormal unit vectors.
    if not (
        np.isclose(np.linalg.norm(align_axis), 1.0, atol=eps)
        and np.isclose(np.linalg.norm(appr_axis), 1.0, atol=eps)
    ):
        raise ValueError("align_axis and appr_axis must be unit vectors.")
    if not (np.isclose(np.dot(align_axis, appr_axis), 0.0, atol=eps)):
        raise ValueError("align_axis and appr_axis must be orthogonal.")

    # upfront normalisation of target directions
    align_vec = _safe_normalize(align_vec, eps)
    appr_vec = _safe_normalize(appr_vec, eps)
    align_axis = _safe_normalize(align_axis, eps)
    appr_axis = _safe_normalize(appr_axis, eps)

    T_list = []

    # primary rotation: align align_axis → align_vec
    R1 = _rot_between(align_axis, align_vec, eps)

    for origin in origins:
        # ------------------------------------------------------------------
        # Optional refinement: twist about new align_vec so
        # (R1·appr_axis) lines up with appr_vec as much as possible.
        # ------------------------------------------------------------------
        if refine:
            new_appr_axis = R1.apply(appr_axis)

            # project both into the plane ⟂ align_vec
            proj_a = new_appr_axis - np.dot(new_appr_axis, align_vec) * align_vec
            proj_b = appr_vec - np.dot(appr_vec, align_vec) * align_vec

            n_a = np.linalg.norm(proj_a)
            n_b = np.linalg.norm(proj_b)

            if n_a > eps and n_b > eps:
                proj_a /= n_a
                proj_b /= n_b

                dot = np.clip(np.dot(proj_a, proj_b), -1.0, 1.0)
                angle = np.arccos(dot)

                # choose sign so the rotation has the shorter direction
                if np.dot(np.cross(proj_a, proj_b), align_vec) < 0:
                    angle = -angle

                R2 = R.from_rotvec(align_vec * angle)
                R_comb = R2 * R1
            else:
                # projections degenerate ⇒ skip refinement
                R_comb = R1
        else:
            R_comb = R1

        # optional re-orthonormalisation to kill tiny drift
        if reortho:
            U, _, Vt = np.linalg.svd(R_comb.as_matrix())
            R_comb = R.from_matrix(U @ Vt)

        # build homogeneous transform
        T = np.eye(4)
        T[:3, :3] = R_comb.as_matrix()
        T[:3, 3] = origin
        T_list.append(T)

    return T_list


def pose_difference(tf_1, tf_2):
    """
    Compute pairwise translation and rotation differences between two sets of transformation matrices.

    Args:
        tf_1: (N, 4, 4) numpy array
        tf_2: (M, 4, 4) numpy array

    Returns:
        translation_diff: (N, M) numpy array of L2 translation differences
        rotation_diff: (N, M) numpy array of rotation angle differences (in radians)
    """
    assert tf_1.shape[1:] == (4, 4) and tf_2.shape[1:] == (
        4,
        4,
    ), "Input tensors must be of shape (N, 4, 4) and (M, 4, 4)"
    N, M = tf_1.shape[0], tf_2.shape[0]

    # Extract translations
    trans_1 = tf_1[:, :3, 3]  # (N, 3)
    trans_2 = tf_2[:, :3, 3]  # (M, 3)

    # Compute translation differences (N, M)
    trans_diff = np.linalg.norm(trans_1[:, None, :] - trans_2[None, :, :], axis=-1)

    # Extract rotations
    rot_1 = tf_1[:, :3, :3]  # (N, 3, 3)
    rot_2 = tf_2[:, :3, :3]  # (M, 3, 3)

    # Compute relative rotation matrices: R_rel = R2 @ R1.T for all pairs
    rot_1_T = rot_1.transpose(0, 2, 1)  # (N, 3, 3)
    rel_rot = np.einsum("mij,njk->nmik", rot_2, rot_1_T)

    # Flatten to compute rotation differences using scipy
    rel_rot_flat = rel_rot.reshape(-1, 3, 3)
    relative_rotations = R.from_matrix(rel_rot_flat)
    rot_diff = np.linalg.norm(relative_rotations.as_rotvec(), axis=1).reshape(N, M)

    return trans_diff, rot_diff


# 2D <-> 3D
def find_center_point(pixels):
    pixels_array = np.array(pixels)

    # Calculate the centroid by averaging the x and y coordinates
    centroid = np.mean(pixels_array, axis=0)

    # Find the pixel that is closest to the centroid
    distances = np.linalg.norm(pixels_array - centroid, axis=1)
    closest_pixel_index = np.argmin(distances)

    # The pixel closest to the centroid
    closest_pixel = pixels_array[closest_pixel_index]

    # Return the closest pixel as an np array with shape N x 2
    return closest_pixel[0], closest_pixel[1]


def direction_on_image_to_vec_in_world(
    angle_rad: float, intrinsic: np.ndarray, extrinsic: np.ndarray
) -> np.ndarray:
    """
    Convert a 2D edge or gradient orientation (in image pixels) into a 3D unit
    viewing direction vector in world coordinates.
    Args:
        angle_rad:      Orientation on the image plane, in radians.
        intrinsic:      3x3 camera intrinsic matrix.
        extrinsic:      4x4 world→camera transform matrix.

    Returns:
        A length-3 unit vector in world coordinates indicating the 3D direction
        corresponding to the input image-plane angle.
    """
    # 1) Build a 2D direction on the normalized image plane (Z=0 plane in camera frame)
    dir_2d = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=float)

    # 2) Compensate for focal lengths to get a camera-frame direction
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    dir_cam = np.array([dir_2d[0] / fx, dir_2d[1] / fy, 0.0], dtype=float)
    dir_cam /= np.linalg.norm(dir_cam)  # normalize in camera space

    # 3) Transform direction into world coordinates (append 0 for homogeneous direction)
    cam2world = np.linalg.inv(extrinsic)
    dir_cam_h = np.append(dir_cam, 0.0)
    dir_world_h = cam2world @ dir_cam_h
    dir_world = dir_world_h[:3]
    dir_world /= np.linalg.norm(dir_world)  # re-normalize to unit length

    # 4) Flip to point “into” the scene (optional, depends on convention)
    return -dir_world


# Image Processing
def compute_gradient(
    image: np.ndarray, kernel_size: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute horizontal and vertical gradients of a mono image using the Sobel operator.

    Args:
        image (np.ndarray): 2D image.
        kernel_size (int): Size of the extended Sobel kernel; must be odd and positive.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Horizontal (x) and vertical (y) gradients.
    """
    # Validate inputs
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd integer.")

    image = np.asarray(image, dtype=np.float64)

    grad_x = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=kernel_size)
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=kernel_size)

    return grad_x, grad_y


def grad_mag_and_direct_from_gradmap(
    grad_x: np.ndarray, grad_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the magnitude and direction (in degrees) of the gradient.

    Args:
        grad_x (np.ndarray): Horizontal (x) gradient.
        grad_y (np.ndarray): Vertical (y) gradient.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Gradient magnitude and direction in degrees.
    """
    if grad_x.shape != grad_y.shape:
        raise ValueError("grad_x and grad_y must have the same shape.")

    magnitude = np.hypot(
        grad_x, grad_y
    )  # more stable and preferred over sqrt(x**2 + y**2)
    direction = np.degrees(np.arctan2(grad_y, grad_x))  # converts to degrees directly

    return magnitude, direction


def get_discontinuity_mask(magnitude, threshold):
    """
    Args:
    magnitude (numpy array): The magnitude of the gradient.
    threshold (float): Threshold value to detect edges.

    Returns:
    numpy array: A binary mask where discontinuities are marked.
    """
    discontinuity_mask = (magnitude > threshold).astype(np.uint8)
    return discontinuity_mask


def grad_mag_and_direct_from_depth(depth_image):
    """
    Args:
        depth_image (np.ndarray): 2D array representing the depth image.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Gradient magnitude and direction.
    """
    grad_x, grad_y = compute_gradient(depth_image)
    magnitude, direction = grad_mag_and_direct_from_gradmap(grad_x, grad_y)
    return magnitude, direction


def get_depth_discontinuity(depth_image, threshold):
    """
    Args:
        depth_image (np.ndarray): 2D array representing the depth image.
        threshold (float): Threshold value to detect edges.
    Returns:
        np.ndarray: A binary mask where discontinuities are marked.
    """
    if not isinstance(depth_image, np.ndarray):
        raise ValueError("depth_image must be a numpy array.")
    magnitude, _ = grad_mag_and_direct_from_depth(depth_image)
    discontinuity_mask = get_discontinuity_mask(magnitude, threshold)
    return discontinuity_mask


def avg_depth_from_bin_mask(
    bin_mask: np.ndarray,
    depth: np.ndarray,
    gradient_direction: np.ndarray,
    gradient_magnitude: np.ndarray,
    skip_close_points: float = 0,
    step_size_min: float = 2.5,
    step_size_max: float = 4.0,
    near_thr: float = 0.1,
    far_thr: float = 10.0,
    depth_diff_low: float = 0.05,
    depth_diff_high: float = 10.0,
) -> np.ndarray:
    """
    Calculate average depth from a binary mask, depth image, gradient direction, and gradient magnitude.
    Args:
        bin_mask (np.ndarray): Binary mask indicating valid pixels.
        depth (np.ndarray): Depth image.
        gradient_direction (np.ndarray): Gradient direction in radians.
        gradient_magnitude (np.ndarray): Gradient magnitude.
        skip_close_points (float): Minimum distance to skip close points.
        step_size_min (float): Minimum step size for depth calculation.
        step_size_max (float): Maximum step size for depth calculation.
        near_thr (float): Near threshold for depth clipping.
        far_thr (float): Far threshold for depth clipping.
        depth_diff_low (float): Lower bound for depth difference to consider valid.
        depth_diff_high (float): Upper bound for depth difference to consider valid.
    Returns:
        np.ndarray: Average depth image.
    """
    assert (
        bin_mask.shape
        == depth.shape
        == gradient_direction.shape
        == gradient_magnitude.shape
    ), "bin_mask, depth, gradient_direction and gradient_magnitude must have the same shape"
    H, W = bin_mask.shape

    avg_depth = np.zeros_like(depth, dtype=np.float32)
    step_size_min = np.clip(step_size_min, 0.1, 10.0)
    step_size_max = np.clip(step_size_max, step_size_min, 10.0)
    near_thr = np.clip(near_thr, 0.01, 2.0)
    far_thr = np.clip(far_thr, near_thr, 10.0)

    for i in range(H):
        for j in range(W):
            if bin_mask[i, j] == 0:
                continue

            magnitude = gradient_magnitude[i, j]
            if not np.isfinite(magnitude) or magnitude > 1e5:
                continue

            direction = gradient_direction[i, j]
            depth_val = np.clip(depth[i, j], near_thr, far_thr)
            step_size = np.clip(1.0 / depth_val, step_size_min, step_size_max)
            di = int(np.round(np.sin(direction) * step_size))
            dj = int(np.round(np.cos(direction) * step_size))
            i_pos = np.clip(i + di, 0, H - 1)
            j_pos = np.clip(j + dj, 0, W - 1)
            i_neg = np.clip(i - di, 0, H - 1)
            j_neg = np.clip(j - dj, 0, W - 1)

            fg_depth = depth[i_neg, j_neg]
            bg_depth = depth[i_pos, j_pos]
            if skip_close_points > 0 and fg_depth < skip_close_points:
                continue

            depth_diff = abs(fg_depth - bg_depth)
            if depth_diff_low < depth_diff < depth_diff_high:
                avg_depth[i, j] = 0.5 * (fg_depth + bg_depth)

    return avg_depth
