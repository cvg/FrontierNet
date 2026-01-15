"""
Replay Visualization for FrontierNet Exploration.

This script replays exploration trajectories from JSON files saved by demo_exploration.py.
It visualizes robot poses and frontiers exactly as they were during exploration.

Usage:
    python replay_visualization.py --mesh examples/mv2HUxq3B53.glb --json_file output/exploration_state.json
"""
import sys
import time
import json
import threading
import logging
import argparse
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import open3d as o3d

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.vis_utils import (
    get_vis_state,
    set_vis_cam_ex,
    set_vis_cam_intr,
    camera_vis_with_cylinders,
    create_camera,
    create_interactive_vis,
    load_mesh,
    register_basic_callbacks,
)
from utils.frontier_utils import read_config_yaml

# Frontier classes for reconstruction
from frontier.frontier import Frontier


def read_exploration_entries(file_path: str) -> List[Dict[str, Any]]:
    """
    Read exploration state entries from JSON lines file.
    
    Args:
        file_path: Path to the JSON lines file.
        
    Returns:
        List of entry dictionaries.
    """
    entries = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    entries.append(entry)
        logging.info(f"Loaded {len(entries)} entries from {file_path}")
        return entries
    except Exception as e:
        logging.error(f"Failed to read file {file_path}: {e}")
        return []


class ReplayApp:
    """
    Replay visualization for exploration trajectories.
    Reads JSON files saved by demo_exploration.py and visualizes 
    robot poses and frontiers step by step.
    """

    # ---------- constants / defaults ----------
    REFRESH_RATE = 50  # Hz
    VOX_SIZE = 0.1

    # Camera-1 (observer) defaults
    CAM1_H, CAM1_W, CAM1_F = 960, 1280, 700.0
    # Camera-2 (robot) defaults
    CAM2_H, CAM2_W, CAM2_F = 480, 480, 300.0

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Config
        self.config = read_config_yaml(args.config)

        # Visualization windows
        self.vis_1 = None  # observer view
        self.vis_2 = None  # robot view

        # Geometry caches (for visualization in o3d)
        self.geometry_vis_1: List[o3d.geometry.Geometry] = []
        self.ft_geometry_vis_1: List[o3d.geometry.Geometry] = []

        # Replay state
        self.entries: List[Dict[str, Any]] = []
        self.current_step: int = 0
        self.is_playing: bool = False
        self.play_speed: float = args.play_speed  # seconds per step
        
        # Current state from JSON
        self.current_robot_poses: Dict[str, np.ndarray] = {}
        self.current_frontiers: List[Dict[str, Any]] = []
        self.current_goal_pose: Optional[np.ndarray] = None
        self.current_robot_id: Optional[int] = None

        # Track last robot pose for frustum updates
        self.last_W_T_C2: np.ndarray = np.eye(4)

        # Track trajectory cylinders (persistent across steps)
        self.trajectory_cylinders: List[o3d.geometry.Geometry] = []

        # Store initial camera extrinsics for vis_1 (observer view)
        self.initial_vis1_extrinsic: Optional[np.ndarray] = None

        # mutex for vis updates
        self._lock = threading.Lock()

    # ---------- visualization helpers ----------

    def update_vis(self) -> None:
        """Update both viewers while preserving their camera extrinsics."""
        if self.vis_1 is None or self.vis_2 is None:
            return

        cam_ex_1 = get_vis_state(self.vis_1)["cam_extrinsic"]
        cam_ex_2 = get_vis_state(self.vis_2)["cam_extrinsic"]

        # Render and restore cams
        self.vis_1.update_renderer()
        self.vis_2.update_renderer()
        set_vis_cam_ex(self.vis_1, cam_ex_1)
        set_vis_cam_ex(self.vis_2, cam_ex_2)

    def update_geometry_vis_1(self) -> None:
        """
        Update o3d geoms overlays in vis_1.
        """
        if self.vis_1 is None or self.vis_2 is None:
            return

        # Save vis_1 camera extrinsic to restore later
        vis1_cam_ex = get_vis_state(self.vis_1)["cam_extrinsic"].copy()

        # vis_2 camera poses
        C2_T_W = get_vis_state(self.vis_2)["cam_extrinsic"]
        W_T_C2 = np.linalg.inv(C2_T_W)

        intr = get_vis_state(self.vis_2)["cam_intrinsic"]
        W2, H2 = intr.width, intr.height
        fx2 = intr.intrinsic_matrix[0, 0]

        wh_ratio = W2 / H2
        fovx_deg = 2.0 * np.degrees(np.arctan(W2 / (2.0 * fx2)))

        frustum_meshes = camera_vis_with_cylinders(
            W_T_C2,
            wh_ratio=wh_ratio,
            scale=0.8,
            weight=0.0,
            color=[0, 0, 1],
            fovx=fovx_deg,
            radius=0.04,
        )

        # Clear previous
        for g in self.geometry_vis_1:
            self.vis_1.remove_geometry(g)
        self.geometry_vis_1.clear()

        # Robot trajectory overlay 
        self.update_trajectory_cylinders(self.vis_1)

        # Frontiers overlay
        self.visualize_all_frontiers(self.vis_1)

        # Frustum overlay
        for g in frustum_meshes:
            self.geometry_vis_1.append(g)
            self.vis_1.add_geometry(g, reset_bounding_box=False)

        if self.args.vis_graph:
            # (optional) topo-graph overlay from current entry
            self._draw_graph_overlay()

        # Restore vis_1 camera to initial position
        if self.initial_vis1_extrinsic is not None:
            set_vis_cam_ex(self.vis_1, self.initial_vis1_extrinsic)
        else:
            # Fallback: restore to what it was before this update
            set_vis_cam_ex(self.vis_1, vis1_cam_ex)

    def _draw_graph_overlay(self) -> None:
        """Draw graph edges from current entry."""
        if not self.entries or self.current_step >= len(self.entries):
            return

        entry = self.entries[self.current_step]
        graph_data = entry.get('graph', {})
        edges = graph_data.get('edges', [])
        nodes = graph_data.get('nodes', [])

        if not edges or not nodes:
            return

        # Build node ID -> position map
        node_positions = {}

        # Process all nodes
        for node in nodes:
            if len(node) < 2:
                continue
            node_id = node[0]
            node_attrs = node[1]
            node_type = node_attrs.get('type')

            if node_type == 'R' and str(node_id) in self.current_robot_poses:
                # Robot node
                pose = self.current_robot_poses[str(node_id)]
                node_positions[node_id] = pose[:3, 3]
            elif node_type == 'F':
                # Frontier node
                ft_by_id = {ft.get('id'): ft for ft in self.current_frontiers}
                if node_id in ft_by_id:
                    pos = ft_by_id[node_id].get('3d_pos')
                    if pos:
                        node_positions[node_id] = np.array(pos, dtype=np.float64)

        # Draw edges
        for edge in edges:
            if len(edge) < 2:
                continue
            src = edge[0]
            dst = edge[1]
            if src in node_positions and dst in node_positions:
                pos1 = node_positions[src]
                pos2 = node_positions[dst]
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector([pos1, pos2])
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector([[1, 1, 1]]) # white
                self.geometry_vis_1.append(line)
                self.vis_1.add_geometry(line, reset_bounding_box=False)

        logging.debug(f"Drew {len(edges)} graph edges connecting {len(node_positions)} nodes")

    # ---------- replay logic ----------

    def load_entries(self) -> bool:
        """Load exploration entries from JSON file."""
        self.entries = read_exploration_entries(self.args.json_file)
        if self.entries:
            logging.info(f"Loaded {len(self.entries)} replay entries.")
            return True
        return False

    def replay(self) -> None:
        """
        Main replay loop - plays through all entries.
        """
        if not self.entries:
            logging.error("No entries to replay.")
            return

        logging.info(f"Starting replay with {len(self.entries)} steps...")
        self.is_playing = True

        for step_idx in range(len(self.entries)):
            if not self.is_playing:
                logging.info("Replay stopped.")
                break

            self.goto_step(step_idx)
            
            # Update display
            self.vis_1.poll_events()
            self.vis_2.poll_events()
            self.vis_1.update_renderer()
            self.vis_2.update_renderer()
            
            time.sleep(self.play_speed)

        self.is_playing = False
        logging.info("Replay finished.")

    def goto_step(self, step: int) -> None:
        """
        Go to a specific step and update visualization.
        """
        if not self.entries:
            return

        step = max(0, min(step, len(self.entries) - 1))
        self.current_step = step
        entry = self.entries[step]

        # Extract data from entry
        robot_poses_dict = entry.get('robot_poses', {})
        self.current_robot_poses = {
            k: np.array(v, dtype=np.float64) for k, v in robot_poses_dict.items()
        }
        self.current_frontiers = entry.get('valid_frontiers', [])
        self.current_robot_id = entry.get('current_robot_id')

        goal_pose_data = entry.get('current_goal_pose')
        if goal_pose_data is not None:
            self.current_goal_pose = np.array(goal_pose_data, dtype=np.float64)
        else:
            self.current_goal_pose = None

        # Update robot camera (vis_2) to current robot pose
        # The current_robot_id points to the last/most recent robot pose in this step
        W_T_C2 = None
        if self.current_robot_id is not None:
            # current_robot_id might be the next ID, so we want the latest pose that exists
            robot_key = str(self.current_robot_id - 1) if str(self.current_robot_id) not in self.current_robot_poses else str(self.current_robot_id)
            if robot_key in self.current_robot_poses:
                W_T_C2 = self.current_robot_poses[robot_key]
            elif len(self.current_robot_poses) > 0:
                # Fallback: use the last pose in the dict
                last_key = sorted(self.current_robot_poses.keys(), key=lambda x: int(x))[-1]
                W_T_C2 = self.current_robot_poses[last_key]

        if W_T_C2 is None and len(self.current_robot_poses) > 0:
            # Final fallback: use the last available pose
            last_key = sorted(self.current_robot_poses.keys(), key=lambda x: int(x))[-1]
            W_T_C2 = self.current_robot_poses[last_key]

        # Apply the camera pose to vis_2 (ego view)
        if W_T_C2 is not None:
            C2_T_W = np.linalg.inv(W_T_C2)
            set_vis_cam_ex(self.vis_2, C2_T_W)
            self.last_W_T_C2 = W_T_C2  # Store for geometry updates
            logging.debug(f"Updated robot camera to pose: {W_T_C2[:3, 3]}")
        else:
            logging.warning(f"No valid robot pose found for step {step + 1}")

        # Update geometry overlays (including camera frustum in vis_1)
        self.update_geometry_vis_1()

        logging.info(f"Step {step + 1}/{len(self.entries)} - "
                     f"Robots: {len(self.current_robot_poses)}, "
                     f"Frontiers: {len(self.current_frontiers)}")

    def next_step(self) -> None:
        """Go to next step."""
        if self.current_step < len(self.entries) - 1:
            self.goto_step(self.current_step + 1)
        else:
            logging.info("Already at the last step.")

    def prev_step(self) -> None:
        """Go to previous step."""
        if self.current_step > 0:
            self.goto_step(self.current_step - 1)
        else:
            logging.info("Already at the first step.")

    # ---------- visualization: frontiers ----------

    def visualize_all_frontiers(self, vis) -> None:
        """
        Overlay all valid frontiers in vis_1 (frustums + axes + goal marker).
        Uses data from self.current_frontiers loaded from JSON.
        """
        C_T_W = get_vis_state(vis)["cam_extrinsic"]  # stash/restore

        for g in self.ft_geometry_vis_1:
            vis.remove_geometry(g)
        self.ft_geometry_vis_1.clear()

        if len(self.current_frontiers) == 0:
            logging.debug("No frontiers to visualize.")
            set_vis_cam_ex(vis, C_T_W)
            return

        # Get current goal frontier ID
        current_goal_id = None
        if self.current_goal_pose is not None and not self.entries:
            # Try to find goal frontier by matching pose
            for ft_data in self.current_frontiers:
                pos3d = ft_data.get('3d_pos')
                if pos3d is not None:
                    pos3d = np.array(pos3d, dtype=np.float64)
                    if np.allclose(pos3d, self.current_goal_pose[:3, 3], atol=0.1):
                        current_goal_id = ft_data.get('id')
                        break
        elif self.entries and self.current_step < len(self.entries):
            # Get goal ID from current entry
            entry = self.entries[self.current_step]
            current_goal_id = entry.get('current_ft_goal_id')

        # Draw each frontier frustum
        for ft_data in self.current_frontiers:
            pos3d = ft_data.get('3d_pos')
            vd = ft_data.get('vd')
            u_gain = ft_data.get('u_gain', 0) or 0
            ft_id = ft_data.get('id')

            if pos3d is None:
                continue

            pos3d = np.array(pos3d, dtype=np.float64)

            # Compute pose matrix from position and view direction
            if vd is not None:
                vd = np.array(vd, dtype=np.float64)
                vd = vd / (np.linalg.norm(vd) + 1e-8)
                W_T_C = self._compute_pose_from_position_direction(pos3d, vd)
            else:
                W_T_C = np.eye(4)
                W_T_C[:3, 3] = pos3d

            # Enlarge goal frontier
            is_goal = (ft_id == current_goal_id)
            scale = 1.2 if is_goal else 0.7
            axis_size = 1.2 if is_goal else 0.7

            frustum = camera_vis_with_cylinders(
                W_T_C,
                wh_ratio=self.CAM2_W / self.CAM2_H,
                scale=scale,
                weight=u_gain / 20.0,
                fovx=2 * np.degrees(np.arctan(self.CAM2_W / (2 * self.CAM2_F))),
                radius=0.04,
                return_mesh=False,
            )
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=axis_size, origin=[0, 0, 0]
            )
            axis.transform(W_T_C)
            frustum.append(axis)
            for g in frustum:
                self.ft_geometry_vis_1.append(g)
                vis.add_geometry(g, reset_bounding_box=False)

        set_vis_cam_ex(vis, C_T_W)

    def _compute_pose_from_position_direction(self, position: np.ndarray, 
                                               direction: np.ndarray) -> np.ndarray:
        """Compute 4x4 pose matrix from position and view direction."""
        # Ensure direction is normalized
        z_axis = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Compute perpendicular axes
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_axis, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        
        # Build pose matrix
        pose = np.eye(4)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = position
        
        return pose

    def update_trajectory_cylinders(self, vis) -> None:
        """
        Update trajectory cylinders based on current step.
        Adds/removes cylinders as needed when navigating forward/backward.
        """
        if not self.current_robot_poses or len(self.current_robot_poses) < 2:
            return

        # Sort robot poses by ID to get trajectory order
        sorted_ids = sorted(self.current_robot_poses.keys(), key=lambda x: int(x))
        num_poses = len(sorted_ids)
        num_segments = num_poses - 1

        # Current number of trajectory cylinders
        current_cylinders = len(self.trajectory_cylinders)

        # Add new cylinders if we moved forward
        if num_segments > current_cylinders:
            for i in range(current_cylinders, num_segments):
                pose1 = self.current_robot_poses[sorted_ids[i]]
                pose2 = self.current_robot_poses[sorted_ids[i + 1]]

                pos1 = pose1[:3, 3]
                pos2 = pose2[:3, 3]

                # Compute vector between poses
                vec = pos2 - pos1
                length = np.linalg.norm(vec)

                if length < 1e-6:
                    # Add None placeholder for zero-length segments
                    self.trajectory_cylinders.append(None)
                    continue

                # Create cylinder oriented along z-axis
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=0.075, height=length, resolution=20, split=4
                )

                yellow_color = [0.0, 0.0, 1.0]  # Bright blue
                cylinder.paint_uniform_color(yellow_color)

                # Compute vertex normals
                cylinder.compute_vertex_normals()

                # Compute rotation to align cylinder with the trajectory vector
                z_axis = np.array([0.0, 0.0, 1.0])
                axis = np.cross(z_axis, vec)
                axis_len = np.linalg.norm(axis)

                if axis_len > 1e-6:
                    axis = axis / axis_len
                    angle = np.arccos(np.clip(np.dot(z_axis, vec) / length, -1.0, 1.0))
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                    cylinder.rotate(R, center=(0, 0, 0))

                # Translate cylinder to midpoint
                midpoint = (pos1 + pos2) / 2.0
                cylinder.translate(midpoint)

                # Add to visualization
                vis.add_geometry(cylinder, reset_bounding_box=False)
                self.trajectory_cylinders.append(cylinder)

        # Remove cylinders if we moved backward
        elif num_segments < current_cylinders:
            for i in range(num_segments, current_cylinders):
                cylinder = self.trajectory_cylinders[i]
                if cylinder is not None:
                    vis.remove_geometry(cylinder)
            # Trim the list
            self.trajectory_cylinders = self.trajectory_cylinders[:num_segments]

    # ---------- setup ----------

    def setup_viewers(self) -> None:
        """Create viewers and add the scene mesh & initial overlays."""
        # Logging level
        if self.args.log_level < 10:
            logging.getLogger().setLevel(logging.NOTSET)
        elif self.args.log_level < 20:
            logging.getLogger().setLevel(logging.DEBUG)
        elif self.args.log_level < 30:
            logging.getLogger().setLevel(logging.INFO)
        elif self.args.log_level < 40:
            logging.getLogger().setLevel(logging.WARNING)
        elif self.args.log_level < 50:
            logging.getLogger().setLevel(logging.ERROR)
        else:
            logging.getLogger().setLevel(logging.CRITICAL)

        # Cameras
        cam_intr_1 = create_camera(self.CAM1_H, self.CAM1_W, self.CAM1_F)
        cam_intr_2 = create_camera(self.CAM2_H, self.CAM2_W, self.CAM2_F)

        self.vis_1 = create_interactive_vis(
            self.CAM1_H,
            self.CAM1_W,
            cam_intr_1,
            show_back_face=False,
            light_on=False,
            z_near=0.02,
            z_far=50.0,
        )
        self.vis_2 = create_interactive_vis(
            self.CAM2_H,
            self.CAM2_W,
            cam_intr_2,
            show_back_face=True,
            light_on=False,
            z_near=0.02,
            z_far=50.0,
        )

        # Load scene mesh
        scene_mesh = load_mesh(self.args.mesh)
        self.vis_1.add_geometry(scene_mesh, reset_bounding_box=False)
        self.vis_2.add_geometry(scene_mesh, reset_bounding_box=True)

        # Initial frustum of vis_2, drawn in vis_1
        C2_T_W = get_vis_state(self.vis_2)["cam_extrinsic"]  # (W→C2)
        frustum = camera_vis_with_cylinders(
            C2_T_W,
            wh_ratio=self.CAM2_W / self.CAM2_H,
            scale=0.8,
            weight=0.0,
            fovx=2 * np.degrees(np.arctan(self.CAM2_W / (2 * self.CAM2_F))),
            radius=0.04,
        )
        for g in frustum:
            self.geometry_vis_1.append(g)
            self.vis_1.add_geometry(g)

        # Global axis
        world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        self.vis_1.add_geometry(world_axis, reset_bounding_box=False)

        # Register basic callbacks
        register_basic_callbacks(self.vis_1)
        register_basic_callbacks(self.vis_2)

        # Reset intrinsics
        set_vis_cam_intr(self.vis_1, cam_intr_1)
        set_vis_cam_intr(self.vis_2, cam_intr_2)

        # Observer & robot poses from config
        obs_C_T_W = np.asarray(self.config["observer_cam_extrinsic"], dtype=float)
        rob_C_T_W = np.asarray(self.config["initial_cam_extrinsic"], dtype=float)
        set_vis_cam_ex(self.vis_1, obs_C_T_W)
        set_vis_cam_ex(self.vis_2, rob_C_T_W)

        # Store initial vis_1 camera extrinsic to keep it fixed
        self.initial_vis1_extrinsic = obs_C_T_W.copy()

    def setup_system(self) -> None:
        """Load replay entries from JSON file."""
        if not self.load_entries():
            logging.error("Failed to load replay entries. Exiting.")
            return
        
        logging.info(f"Replay system ready with {len(self.entries)} steps.")

    def run(self) -> None:
        self.setup_viewers()
        self.setup_system()

        if not self.entries:
            logging.error("No entries loaded, cannot start.")
            return

        # Go to first step
        self.goto_step(0)

        if self.args.auto_start:
            self.replay()
        else:
            logging.info(
                " --- Replay mode: Use N/Right for next, P/Left for prev, SPACE to auto-play ---"
            )

            def on_space(vis):
                if self.is_playing:
                    self.is_playing = False
                    logging.info("Replay paused.")
                else:
                    self.replay()
                return False

            def on_next(vis):
                self.next_step()
                return False

            def on_prev(vis):
                self.prev_step()
                return False

            # Register callbacks: Space, N, P, Right Arrow, Left Arrow
            self.vis_1.register_key_callback(32, on_space)  # Space
            self.vis_2.register_key_callback(32, on_space)
            self.vis_1.register_key_callback(ord('N'), on_next)
            self.vis_2.register_key_callback(ord('N'), on_next)
            self.vis_1.register_key_callback(ord('P'), on_prev)
            self.vis_2.register_key_callback(ord('P'), on_prev)
            self.vis_1.register_key_callback(262, on_next)  # Right arrow
            self.vis_2.register_key_callback(262, on_next)
            self.vis_1.register_key_callback(263, on_prev)  # Left arrow
            self.vis_2.register_key_callback(263, on_prev)

            logging.info(" --- PRESS SPACE TO START REPLAY, N/P or Arrow keys to step ---")

        try:
            while self.vis_1.poll_events() and self.vis_2.poll_events():
                with self._lock:
                    self.update_vis()
                time.sleep(1.0 / self.REFRESH_RATE)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.vis_1.destroy_window()
            self.vis_2.destroy_window()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay visualization for FrontierNet exploration trajectories"
    )
    p.add_argument("--mesh", type=str, required=True, help="Path to the mesh file")
    p.add_argument(
        "--json_file", "-j",
        type=str,
        required=True,
        help="Path to the exploration state JSON file to replay"
    )
    p.add_argument(
        "--config",
        type=str,
        default="config/hm3d_exploration.yaml",
        help="Configuration file (for camera settings)",
    )
    p.add_argument(
        "--auto_start",
        action="store_true",
        default=False,
        help="Auto-start replay loop",
    )
    p.add_argument(
        "--play_speed",
        type=float,
        default=0.5,
        help="Seconds per step during auto-replay (default: 0.5)",
    )
    p.add_argument(
        "--vis_graph",
        action="store_true",
        default=False,
        help="Visualize the topological graph edges",
    )
    p.add_argument(
        "--log_level",
        "-ll",
        type=int,
        default=20,
        help="logging level (0=notset, 10=debug, 20=info...)",
    )
    return p


def main():
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.WARNING,
    )
    print(f"Open3D version: {o3d.__version__}")

    parser = build_arg_parser()
    args = parser.parse_args()

    app = ReplayApp(args)
    app.run()


if __name__ == "__main__":
    main()
