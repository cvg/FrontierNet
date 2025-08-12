import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from typing import List, Optional, Tuple, Any
from utils.geometry import compute_alignment_transforms, pose_difference, rot2quat


## Open3D Basic ##
def create_camera(H, W, focal):
    fx, fy = focal, focal
    cx, cy = W / 2.0 - 0.5, H / 2.0 - 0.5
    return o3d.camera.PinholeCameraIntrinsic(
        width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy
    )


def set_viewpoint_ctr(vis, z_near=0.02, z_far=15.0):

    ctr = vis.get_view_control()
    ctr.set_constant_z_far(z_far)
    ctr.set_constant_z_near(z_near)
    return ctr


def create_interactive_vis(
    H,
    W,
    camera_intrinsic,
    camera_extrinsic=np.eye(4),
    show_back_face=False,
    light_on=False,
    z_near=0.02,
    z_far=15.0,
):

    vis = o3d.visualization.VisualizerWithKeyCallback()
    assert H == camera_intrinsic.height
    assert W == camera_intrinsic.width
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = show_back_face
    vis.get_render_option().light_on = light_on
    ctr = set_viewpoint_ctr(vis, z_near, z_far)
    param = ctr.convert_to_pinhole_camera_parameters()
    param.intrinsic = camera_intrinsic
    param.extrinsic = camera_extrinsic
    success = ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    assert success
    return vis

def is_vis_moving(vis, previous_pose, trans_thre=0.02, rot_thre=0.05):
    state_v2 = get_vis_state(vis)
    cam_pose_v2 = np.linalg.inv(state_v2["cam_extrinsic"])
    trans_diff, rot_diff = pose_difference(
        previous_pose.reshape(1, 4, 4), cam_pose_v2.reshape(1, 4, 4)
    )

    if trans_diff[0, 0] > trans_thre or rot_diff[0, 0] > rot_thre:
        return True
    else:
        return False

def get_vis_state(vis):
    """Get the state of the visualizer
    returns: state dict:
    {
        "cam_intrinsic": PinholeCameraIntrinsic,
        "cam_extrinsic": 4x4 np.array,
    }
    """
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    cam_intrinsic = param.intrinsic
    cam_extrinsic = param.extrinsic
    return {"cam_intrinsic": cam_intrinsic, "cam_extrinsic": cam_extrinsic}


def set_vis_cam_ex(vis, cam_extrinsic):
    # get the original camera intrinsic
    intrinsic = get_vis_state(vis)["cam_intrinsic"]
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    param.extrinsic = cam_extrinsic
    param.intrinsic = intrinsic
    success = ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    assert success
    return


def set_vis_cam_intr(vis, cam_intrinsic):
    """Set the camera intrinsic of the visualizer
    vis: Open3D Visualizer
    cam_intrinsic: PinholeCameraIntrinsic
    """
    extrinsic = get_vis_state(vis)["cam_extrinsic"]
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    param.intrinsic = cam_intrinsic
    param.extrinsic = extrinsic
    success = ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    assert success
    return


## PERCEPTION ##
def capture_rgb(vis, return_rgb_type="np"):
    """Capture the RGB image from the visualizer
    vis: Open3D Visualizer
    return_rgb_type: str, default "np"
    returns: dict [np.array or o3d.geometry.Image, camera intrinsics, extrinsics]
    """
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic_matrix = cam_params.extrinsic.copy()
    intrinsic_matrix = cam_params.intrinsic.intrinsic_matrix

    color_image = vis.capture_screen_float_buffer(True)
    color_image_np = (np.asarray(color_image) * 255).astype(np.uint8)
    color_image = o3d.geometry.Image(color_image_np)

    return_rgb = color_image_np if return_rgb_type == "np" else color_image

    return {
        "image": return_rgb,
        "intrinsics": intrinsic_matrix,
        "extrinsics": extrinsic_matrix,
    }


def capture_depth(vis, return_depth_type="np"):
    """Capture the depth image from the visualizer
    vis: Open3D Visualizer
    return_depth_type: str, default "np"
    returns: dict [np.array or o3d.geometry.Image, camera intrinsics, extrinsics]
    """

    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic_matrix = cam_params.extrinsic.copy()
    intrinsic_matrix = cam_params.intrinsic.intrinsic_matrix

    depth_image = vis.capture_depth_float_buffer(True)
    depth_image_np = np.asarray(depth_image).astype(np.float32)
    depth_image = o3d.geometry.Image((depth_image_np * 255).astype(np.uint16))

    return_depth = depth_image_np if return_depth_type == "np" else depth_image

    return {
        "depth": return_depth,
        "intrinsics": intrinsic_matrix,
        "extrinsics": extrinsic_matrix,
    }


def capture_rgbd_and_visualize(vis):
    rgb = capture_rgb(vis, return_rgb_type="np")
    depth = capture_depth(vis, return_depth_type="np")
    # visualize the rgb and depth side by side, depth, use jet colormap
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rgb["image"])
    ax[0].axis("off")
    ax[0].set_title("RGB")
    ax[1].imshow(depth["depth"], cmap="jet")
    ax[1].axis("off")
    ax[1].set_title("Depth")
    plt.show()
    return False


def capture_rgbd_and_save(vis, save_dir):
    rgb = capture_rgb(vis, return_rgb_type="np")["image"]
    depth = capture_depth(vis, return_depth_type="np")["depth"]
    # save the rgb and depth to the save_dir
    rgb_path = save_dir + "/rgb.png"
    depth_path = save_dir + "/depth.npy"
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    depth_jpg_path = save_dir + "/depth.jpg"
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-6)
    cv2.imwrite(depth_jpg_path, (depth * 255).astype(np.uint8))
    return False


def load_mesh(meshfile):
    mesh = o3d.io.read_triangle_mesh(meshfile, enable_post_processing=True)
    print("load mesh: {}".format(mesh))
    return mesh


def create_cylinder_between_points(
    point1, point2, radius=0.01, resolution=20, color=(0, 0, 0)
):
    """
    Create a cylinder between two 3D points.
    return: An Open3D TriangleMesh representing the cylinder.
    """
    # Compute the vector between the two points
    vec = point2 - point1
    length = np.linalg.norm(vec)
    if length == 0:
        return None

    # Create a cylinder oriented along the z-axis
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=resolution
    )
    cylinder.paint_uniform_color(color)
    cylinder.compute_vertex_normals()

    # Compute rotation to align the cylinder with the vector
    z_axis = np.array([0, 0, 1])  # default cylinder direction
    axis = np.cross(z_axis, vec)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0:
        axis = axis / axis_len
        angle = np.arccos(np.dot(z_axis, vec) / length)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(R, center=(0, 0, 0))

    # Translate the cylinder to its correct position
    midpoint = (point1 + point2) / 2
    cylinder.translate(midpoint)

    return cylinder


def camera_vis_with_cylinders(
    transform: np.ndarray,
    wh_ratio: float = 4.0 / 3.0,
    scale: float = 1.0,
    fovx: float = 90.0,
    weight: float = 1.0,
    color: tuple = None,
    radius=0.01,
    return_mesh=False,
):
    """
    A camera frustum for visualization using cylinders with a yellow-to-red color gradient based on weight.
    transfomr: 4x4 numpy array, camera to world
    return: A list of Open3D geometries representing the camera frustum.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"Transform Matrix must be 4x4, but got {transform.shape}")

    # Compute frustum points
    pw = np.tan(np.deg2rad(fovx / 2.0)) * scale
    ph = pw / wh_ratio
    all_points = np.array(
        [
            [0.0, 0.0, 0.0],  # Frustum apex
            [pw, ph, scale],  # Top right
            [pw, -ph, scale],  # Bottom right
            [-pw, ph, scale],  # Top left
            [-pw, -ph, scale],  # Bottom left
        ]
    )

    # Define frustum edges by connecting points
    line_indices = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],  # Apex to corners
            [1, 2],
            [1, 3],
            [3, 4],
            [2, 4],  # Frustum base edges
        ]
    )

    # Custom yellow-to-red colormap
    yellow_red_cmap = LinearSegmentedColormap.from_list(
        "yellow_red", ["#F6FF55", "#FF0202"]
    )

    if color is not None:
        my_color = color

    else:
        if weight > 0 or weight == 0:
            color_value = min(weight, 1.0)
            my_color = yellow_red_cmap(color_value)[
                :3
            ]  # Get RGB from custom colormap for all cylinders
        else:
            # put blue color for weight = 0
            my_color = [0, 0, 1]

    # Create cylinders for each line segment in the frustum
    cylinders = []
    for start_idx, end_idx in line_indices:
        start_point = all_points[start_idx]
        end_point = all_points[end_idx]

        # Create the cylinder with the same color
        cylinder = create_cylinder_between_points(
            start_point, end_point, radius=radius, color=my_color
        )
        if cylinder is not None:
            cylinders.append(cylinder)

    # Apply the transformation to all cylinders
    for cylinder in cylinders:
        cylinder.transform(transform)

    if return_mesh:
        m = cylinders[0]
        for c in cylinders[1:]:
            m += c
        return m
    return cylinders


def print_camera_params(vis):
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    print("Camera intrinsics:")
    print(cam_params.intrinsic.intrinsic_matrix)
    print("Camera extrinsics:")
    print(cam_params.extrinsic)
    print("extrinsic quaternion")
    rot = cam_params.extrinsic.copy()[:3, :3]
    print(rot2quat(rot))
    return False


def move_forward(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0.1, 0, 0)
    return False


def move_backward(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(-0.1, 0, 0)
    return False


def move_up(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0, 0.1)
    return False


def move_down(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0, -0.1)
    return False


def move_left(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, -0.1, 0)
    return False


def move_right(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0.1, 0)
    return False


def rotate_left(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(-10, 0, 0)
    return False


def rotate_right(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(10, 0, 0)
    return False


def register_basic_callbacks(vis):
    vis.register_key_callback(ord("K"), print_camera_params)
    vis.register_key_callback(ord("W"), move_forward)
    vis.register_key_callback(ord("A"), move_left)
    vis.register_key_callback(ord("D"), move_right)
    vis.register_key_callback(ord("S"), move_backward)
    vis.register_key_callback(ord("Q"), move_up)
    vis.register_key_callback(ord("Z"), move_down)
    vis.register_key_callback(ord("L"), rotate_right)
    vis.register_key_callback(ord("J"), rotate_left)
    vis.register_key_callback(ord("C"), capture_rgbd_and_visualize)
    return vis



## PLOTTING ##
def backproject_rgbd_to_pointcloud(rgb, depth, K, T_cam_to_world):
    """
    Backprojects an RGBD image into a 3D point cloud in world coordinates.

    Args:
        rgb: (H, W, 3) uint8 RGB image
        depth: (H, W) float or uint16 depth image (in meters)
        K: (3, 3) camera intrinsic matrix
        T_cam_to_world: (4, 4) camera-to-world extrinsic matrix

    Returns:
        (N, 6) float32 point cloud in world coordinates [x, y, z, r, g, b]
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create a grid of (u, v) pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    d = depth.flatten()
    color = rgb.reshape(-1, 3)

    # Filter out zero or invalid depth
    valid = d > 0
    u = u[valid]
    v = v[valid]
    d = d[valid]
    color = color[valid]

    # Backproject to camera coordinates
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d
    ones = np.ones_like(z)
    cam_points = np.vstack((x, y, z, ones))  # shape: (4, N)

    # Transform to world coordinates
    world_points = T_cam_to_world @ cam_points  # shape: (4, N)
    world_points = world_points[:3].T  # shape: (N, 3)

    # Combine with color
    colored_points = np.hstack((world_points, color.astype(np.float32)))

    return colored_points


def get_heatmap(values, cmap_name="turbo", invert=False, min_max_values=None):
    if invert:
        values = -values

    if min_max_values is not None:
        values = np.clip(values, min_max_values[0], min_max_values[1])
        values = (values - min_max_values[0]) / (
            min_max_values[1] - min_max_values[0] + 1e-6
        )
    else:
        values = (values - values.min()) / (values.max() - values.min() + 1e-6)
    colormaps = plt.get_cmap(cmap_name)
    rgb = colormaps(values)[..., :3]
    return rgb


def visualize_distance_field(distance, color, cmap_name="turbo", smooth=0.2):
    distance = cv2.resize(
        distance, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    distance = distance**smooth
    dist_vis = get_heatmap(distance, cmap_name=cmap_name)
    dist_vis = (dist_vis * 255).astype(np.uint8)[..., ::-1].copy()
    vis = cv2.addWeighted(color, 0.4, dist_vis, 0.6, 0)
    return vis


def visualize_gain_map(
    gain,
    distance,
    color,
    cmap_name="cool",
    smooth=0.3,
    dilate_kernel=5,
    blur_kernel=3,
    min_max_gain=None,
):
    assert dilate_kernel % 2 == 1, "Dilate kernel size must be odd"
    assert blur_kernel % 2 == 1, "Blur kernel size must be odd"

    gain = cv2.resize(
        gain, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    distance = cv2.resize(
        distance, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    distance = distance**smooth
    gain_mask = gain > 0
    new_gain = np.zeros_like(gain)
    new_gain[gain_mask] = gain[gain_mask]
    new_gain = cv2.dilate(
        new_gain, np.ones((dilate_kernel, dilate_kernel), np.uint8), iterations=4
    )
    new_gain = cv2.blur(new_gain, (blur_kernel, blur_kernel), 0)
    gain_vis = get_heatmap(
        new_gain,
        cmap_name=cmap_name,
        min_max_values=min_max_gain if min_max_gain is not None else None,
    )

    gain_vis = (gain_vis * 255).astype(np.uint8)[..., ::-1].copy()
    vis = cv2.addWeighted(color, 0.4, gain_vis, 0.6, 0)
    return vis


def visualize_3D_frontier(
    ft_3D: List[Any],
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
) -> None:
    """
    Render 3D frontier spheres, their view‐direction meshes,
    the camera frustum, and the RGB-D point cloud.
    """
    if ft_3D is None or (isinstance(ft_3D, list) and len(ft_3D) == 0):
        print("No 3D frontiers to visualize.")
        return

    inv_extrinsic = np.linalg.inv(extrinsic)
    gain_max = 20.0  # max volume for normalizing cylinder weight
    scale = 0.08  # sphere radius

    scene_geometry = []

    # skip if None, empty list, or a 0-d NumPy array holding None
    if (
        ft_3D is None
        or (isinstance(ft_3D, list) and len(ft_3D) == 0)
        or (isinstance(ft_3D, np.ndarray) and ft_3D.shape == ())
    ):
        print("No 3D frontiers to visualize.")
        return

    for ft in ft_3D:
        # 3D sphere at frontier position
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=scale)
        mesh.paint_uniform_color((0, 0, 0))
        mesh.compute_vertex_normals()
        mesh.translate(ft.pos3d)
        scene_geometry.append(mesh)

        # view‐direction cylinder(s) from compute_alignment_transforms
        t_mat = compute_alignment_transforms(
            origins=[ft.pos3d],
            align_vec=ft.view_direction,
            align_axis=[0, 0, 1],
            appr_vec=[0, 0, -1],
            appr_axis=[0, 1, 0],
        )[0]

        cylinders = camera_vis_with_cylinders(
            t_mat,
            wh_ratio=intrinsic[0, 0] / intrinsic[1, 1],
            scale=0.3,
            fovx=90.0,
            weight=ft.gain / gain_max,
            radius=0.025,
            return_mesh=False,
        )
        # add camera coordinate frame at the frontier
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        coord.transform(t_mat)
        scene_geometry.extend(cylinders + [coord])

    # camera frustum
    frustum = camera_vis_with_cylinders(
        inv_extrinsic,
        wh_ratio=intrinsic[0, 0] / intrinsic[1, 1],
        scale=0.4,
        fovx=90.0,
        weight=0.0,
        color=[0, 0, 1],
        radius=0.015,
        return_mesh=False,
    )
    scene_geometry.extend(frustum)

    # backproject RGB-D to point cloud
    pc = backproject_rgbd_to_pointcloud(rgb, depth, intrinsic, inv_extrinsic)
    pc_geometry = o3d.geometry.PointCloud()
    pc_geometry.points = o3d.utility.Vector3dVector(pc[:, :3])
    pc_geometry.colors = o3d.utility.Vector3dVector(
        pc[:, 3:6] / 255.0
    )  # normalized RGB
    scene_geometry.append(pc_geometry)

    o3d.visualization.draw_geometries(scene_geometry)


def visualize_2D_frontier(
    ft_3D: Optional[List[Any]],
    rgb: np.ndarray,
    img_size: Tuple[int, int],
    arrow_color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Overlay 2D frontier points and their view‐directions on the RGB image.
    """
    vis = rgb.copy()

    # skip if None, empty list, or a 0-d NumPy array holding None
    if (
        ft_3D is None
        or (isinstance(ft_3D, list) and len(ft_3D) == 0)
        or (isinstance(ft_3D, np.ndarray) and ft_3D.shape == ())
    ):
        return vis

    H, W = img_size
    length = 55  # arrow length in pixels
    thickness = 4
    tip_scale = 0.4
    default_color = (0, 165, 255)  # BGR

    for ft in ft_3D:
        col = arrow_color if arrow_color and len(arrow_color) == 3 else default_color
        # denormalize pixel position
        u, v = ft.pixel_pos
        px = int(round(u * W))
        py = int(round(v * H))

        # skip out-of-bounds
        if not (0 <= px < vis.shape[1] and 0 <= py < vis.shape[0]):
            continue

        # draw circle
        cv2.circle(vis, (px, py), 6, col, -1, lineType=cv2.LINE_AA)

        # draw arrow for view direction
        angle = ft.direct_angle
        dx = int(np.cos(angle) * length)
        dy = int(np.sin(angle) * length)
        end = (px - dx, py - dy)
        cv2.arrowedLine(
            vis,
            (px, py),
            end,
            col,
            thickness=thickness,
            tipLength=tip_scale,
            line_type=cv2.LINE_AA,
        )

    return vis
