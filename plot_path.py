"""
Interactive path visualization script for FrontierNet.
Loads a mesh, config, and JSON exploration state, then visualizes the robot path
with viridis colormap and cylinders between waypoints in an interactive 3D window.
"""

import argparse
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


def load_mesh(mesh_path: str) -> o3d.geometry.TriangleMesh:
    """Load mesh from file.
    
    Args:
        mesh_path: Path to mesh file (supports .glb, .gltf, .ply, .obj)
        
    Returns:
        TriangleMesh object
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
    print(f"load mesh: {mesh}")
    return mesh


def extract_positions_from_poses(robot_poses: dict) -> list:
    """Extract camera positions from robot poses dictionary.
    
    Args:
        robot_poses: Dictionary mapping robot_id to 4x4 transformation matrix
        
    Returns:
        List of (robot_id, position) tuples, sorted by robot_id
    """
    positions = []
    for robot_id, pose_matrix in robot_poses.items():
        pose = np.array(pose_matrix)
        # Position is in the last column (translation part of 4x4 matrix)
        position = pose[:3, 3]
        positions.append((int(robot_id), position))
    
    # Sort by robot_id to get chronological order
    positions.sort(key=lambda x: x[0])
    return positions


def create_cylinder_between_points(
    point1: np.ndarray, 
    point2: np.ndarray, 
    radius: float,
    color: tuple
) -> o3d.geometry.TriangleMesh:
    """Create a cylinder mesh between two 3D points.
    
    Args:
        point1: Start point (3D)
        point2: End point (3D)
        radius: Cylinder radius
        color: RGB color tuple
        
    Returns:
        TriangleMesh cylinder positioned between the two points
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    
    # Calculate cylinder properties
    direction = point2 - point1
    height = np.linalg.norm(direction)
    
    if height < 1e-6:
        # Return a tiny sphere if points are too close
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color(color)
        return sphere
    
    # Create cylinder centered at origin, aligned with Z-axis
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=height, resolution=20, split=4
    )
    
    # Calculate rotation to align cylinder with direction
    direction_normalized = direction / height
    z_axis = np.array([0, 0, 1])
    
    # Rotation axis and angle
    rotation_axis = np.cross(z_axis, direction_normalized)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    if rotation_axis_norm > 1e-6:
        rotation_axis = rotation_axis / rotation_axis_norm
        cos_angle = np.dot(z_axis, direction_normalized)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Rodrigues rotation
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        cylinder.rotate(R, center=(0, 0, 0))
    elif np.dot(z_axis, direction_normalized) < 0:
        # 180 degree rotation needed
        cylinder.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0]),
            center=(0, 0, 0)
        )
    
    # Translate to midpoint
    midpoint = (point1 + point2) / 2
    cylinder.translate(midpoint)
    
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color(color)
    return cylinder


def create_path_geometries(positions: list, cylinder_radius: float = 0.02) -> list:
    """Create path visualization geometries with viridis colormap.
    
    Args:
        positions: List of (robot_id, position) tuples
        cylinder_radius: Radius of cylinders connecting waypoints
        
    Returns:
        List of Open3D geometry objects
    """
    if len(positions) < 2:
        print("Need at least 2 positions to visualize path")
        return []
    
    geometries = []
    
    # Get viridis colormap
    cmap = plt.cm.viridis
    n_points = len(positions)
    
    # Create spheres at each waypoint
    for i, (robot_id, pos) in enumerate(positions):
        color = cmap(i / (n_points - 1))[:3]  # RGB from viridis
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cylinder_radius * 2)
        sphere.translate(pos)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        geometries.append(sphere)
    
    # Create cylinders between consecutive waypoints
    for i in range(len(positions) - 1):
        _, pos1 = positions[i]
        _, pos2 = positions[i + 1]
        
        # Skip if positions are the same (stationary)
        if np.allclose(pos1, pos2, atol=1e-6):
            continue
        
        # Color based on midpoint in path
        color = cmap((i + 0.5) / (n_points - 1))[:3]
        
        cylinder = create_cylinder_between_points(pos1, pos2, cylinder_radius, color)
        geometries.append(cylinder)
    
    return geometries


def load_yaml_config(yaml_path: str) -> dict:
    """Load YAML configuration file.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        Dictionary containing configuration
    """
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_json_state(json_path: str) -> dict:
    """Load exploration state from JSON file.
    
    Handles both standard JSON and JSONL (JSON Lines) format.
    For JSONL, returns the last valid JSON object (final state).
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary containing exploration state
    """
    with open(json_path, 'r') as f:
        content = f.read().strip()
    
    # Try standard JSON first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Handle JSONL format (multiple JSON objects, one per line)
    # Return the last valid JSON object (final exploration state)
    last_valid = None
    for line in content.split('\n'):
        line = line.strip()
        if line:
            try:
                last_valid = json.loads(line)
            except json.JSONDecodeError:
                continue
    
    if last_valid is None:
        raise ValueError(f"Could not parse any valid JSON from {json_path}")
    
    return last_valid


def main():
    parser = argparse.ArgumentParser(
        description="Interactive path visualization for FrontierNet exploration"
    )
    parser.add_argument(
        "--mesh", "-m",
        type=str,
        required=True,
        help="Path to mesh file (GLB, GLTF, PLY, OBJ)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/hm3d_exploration.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--json", "-j",
        type=str,
        required=True,
        help="Path to JSON exploration state file"
    )
    parser.add_argument(
        "--cylinder-radius", "-r",
        type=float,
        default=0.04,
        help="Radius of path cylinders"
    )
    
    args = parser.parse_args()
    
    print(f"Loading mesh from: {args.mesh}")
    print(f"Loading exploration state from: {args.json}")
    
    # Load JSON state
    state = load_json_state(args.json)
    
    if "robot_poses" not in state:
        print("Error: JSON file does not contain 'robot_poses' key")
        return
    
    robot_poses = state["robot_poses"]
    print(f"Found {len(robot_poses)} robot poses")
    
    # Load config
    config = load_yaml_config(args.config)
    print(f"Loading config from: {args.config}")
    
    # Load mesh
    mesh = load_mesh(args.mesh)
    print("Mesh loaded")
    
    # Extract positions and create path visualization
    positions = extract_positions_from_poses(robot_poses)
    print(f"Extracted {len(positions)} positions for path visualization")
    
    path_geometries = create_path_geometries(positions, cylinder_radius=args.cylinder_radius)
    print(f"Created {len(path_geometries)} path geometry objects")
    
    # Collect all geometries
    all_geometries = [mesh] + path_geometries
    
    # Create interactive visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="FrontierNet Path Visualization", width=1280, height=960)
    
    # Add all geometries
    for geom in all_geometries:
        vis.add_geometry(geom)
    
    # Configure render options - disable backface rendering
    render_opt = vis.get_render_option()
    render_opt.mesh_show_back_face = False
    render_opt.light_on = True
    
    # Set camera if observer_cam_extrinsic is in config
    if "observer_cam_extrinsic" in config:
        extrinsic = np.array(config["observer_cam_extrinsic"])
        ctr = vis.get_view_control()
        
        # Create camera parameters
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic
        
        # Set intrinsic to match window size
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=1280, height=960,
            fx=700.0, fy=700.0,
            cx=639.5, cy=479.5
        )
        param.intrinsic = intrinsic
        
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        print("Camera set from observer_cam_extrinsic")
    
    print("\nInteractive visualization started. Controls:")
    print("  - Left mouse: Rotate")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Q or Esc: Quit")
    
    # Run visualizer
    vis.run()
    vis.destroy_window()
    
    print("Visualization closed")


if __name__ == "__main__":
    main()
