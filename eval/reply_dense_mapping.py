#!/usr/bin/env python3
"""
Replay script for FrontierNet exploration.

This script replays exploration trajectories from JSON files saved by demo_exploration.py,
performs TSDF integration using WaveMapper, and computes the mapped volume at each step.
"""
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import open3d as o3d

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.vis_utils import (
    create_camera,
    create_interactive_vis,
    get_vis_state,
    set_vis_cam_ex,
    set_vis_cam_intr,
    load_mesh,
    capture_depth,
    capture_rgb,
)
from utils.frontier_utils import read_config_yaml
from mapping.wavemap import WaveMapper
from frontier.manager import FrontierManager


def invert_transformation_matrix(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 transformation matrix."""
    return np.linalg.inv(T)


class ReplayApp:
    """
    Replay exploration trajectories and compute mapped volume using WaveMapper.
    """

    # Camera defaults (matching demo_exploration.py)
    CAM_H, CAM_W, CAM_F = 480, 480, 300.0

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = read_config_yaml(args.config)
        
        # Override camera params from config if available
        self.CAM_H = int(self.config.get("cam_height", self.CAM_H))
        self.CAM_W = int(self.config.get("cam_width", self.CAM_W))
        self.CAM_F = float(self.config.get("focal_length", self.CAM_F))
        
        # Visualization
        self.vis: Optional[o3d.visualization.VisualizerWithKeyCallback] = None
        
        # Mapper
        self.mapper: Optional[WaveMapper] = None
        self.voxel_size = float(self.config.get("voxel_size", 0.1))
        
        # Depth range
        self.depth_range = float(self.config.get("depth_range", 3.5))
        
        # Results
        self.results: List[Dict[str, Any]] = []
        
        # Output path (set in run())
        self.output_path: Optional[str] = None

    def setup_viewer(self) -> None:
        """Create visualization window and load mesh."""
        cam_intr = create_camera(self.CAM_H, self.CAM_W, self.CAM_F)
        
        self.vis = create_interactive_vis(
            self.CAM_H,
            self.CAM_W,
            cam_intr,
            show_back_face=True,
            light_on=False,
            z_near=0.02,
            z_far=50.0,
        )
        
        # Load scene mesh
        scene_mesh = load_mesh(self.args.mesh)
        self.vis.add_geometry(scene_mesh, reset_bounding_box=True)
        
        logging.info(f"Loaded mesh from {self.args.mesh}")

    def setup_mapper(self) -> None:
        """Initialize WaveMapper with camera parameters."""
        intr = get_vis_state(self.vis)["cam_intrinsic"]
        
        params = {
            "min_cell_width": self.voxel_size / 2.0,
            "width": intr.width,
            "height": intr.height,
            "fx": intr.intrinsic_matrix[0, 0],
            "fy": intr.intrinsic_matrix[1, 1],
            "cx": intr.intrinsic_matrix[0, 2],
            "cy": intr.intrinsic_matrix[1, 2],
            "min_range": 0.05,
            "max_range": self.depth_range,
            "resolution": self.voxel_size,
        }
        
        self.mapper = WaveMapper(params=params)
        logging.info("WaveMapper initialized")

    def load_trajectory(self) -> List[Dict[str, Any]]:
        """Load trajectory entries from JSON file."""
        entries = FrontierManager.read_from_file(self.args.json_file)
        logging.info(f"Loaded {len(entries)} entries from {self.args.json_file}")
        return entries

    def get_robot_poses_from_entries(self, entries: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Extract robot poses (W_T_C) from trajectory entries."""
        all_poses = []
        seen_ids = set()
        
        for entry in entries:
            robot_poses = entry.get("robot_poses", {})
            for rid, pose_list in robot_poses.items():
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    pose = np.array(pose_list, dtype=np.float64)
                    all_poses.append(pose)
        
        logging.info(f"Extracted {len(all_poses)} unique robot poses")
        return all_poses

    def capture_depth_at_pose(self, W_T_C: np.ndarray) -> np.ndarray:
        """
        Teleport camera to pose and capture depth.
        
        Args:
            W_T_C: 4x4 camera pose in world frame (world to camera transform)
        
        Returns:
            Depth image as numpy array
        """
        # Convert W_T_C to C_T_W (extrinsic) for Open3D
        C_T_W = invert_transformation_matrix(W_T_C)
        
        # Set camera pose
        set_vis_cam_ex(self.vis, C_T_W)
        
        # Update renderer
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.05)  # Small delay to ensure rendering completes
        
        # Capture depth
        depth_data = capture_depth(self.vis, return_depth_type="np")
        depth = depth_data["depth"]
        
        # Clamp depth to range
        depth[depth > self.depth_range] = 0.0
        
        return depth

    def compute_mapped_volume(self, only_free=True) -> float:
        """
        Compute the mapped volume in m^3 from the current occupancy grid.
        
        Returns:
            Volume in cubic meters
        """
        self.mapper.interpolate_occupancy_grid()
        og = self.mapper.get_occupancy_grid()
        
        occ_pts = og.get("occupied")
        if occ_pts is None or len(occ_pts) == 0:
            return 0.0
        
        free_pts = og.get("free")
        if free_pts is not None and len(free_pts) > 0:
            logging.debug(f"Occupied voxels: {len(occ_pts)}, Free voxels: {len(free_pts)}")

        # Each occupied voxel has volume = voxel_size^3
        num_occ_voxels = len(occ_pts)
        num_free_voxels = len(free_pts) if free_pts is not None else 0
        num_voxels = num_free_voxels if only_free else (num_free_voxels + num_occ_voxels)
        volume = num_voxels * (self.voxel_size ** 3)

        return volume

    def replay_trajectory(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Replay the trajectory, perform mapping at each step, and compute volume.
        
        Args:
            entries: List of trajectory entries from JSON
            
        Returns:
            List of entries with added 'mapped_vol' field
        """
        poses = self.get_robot_poses_from_entries(entries)
        
        if len(poses) == 0:
            logging.warning("No poses found in trajectory")
            return entries
        
        max_steps = min(len(poses), self.args.max_steps)
        save_interval = self.args.save_interval
        logging.info(f"Replaying {max_steps} steps (saving every {save_interval} steps)")
        
        updated_entries = []
        
        for i, entry in enumerate(entries):
            # Get poses up to this entry
            robot_poses = entry.get("robot_poses", {})
            current_pose_ids = list(robot_poses.keys())
            
            if i >= max_steps:
                # Beyond max steps, just copy entry without mapping
                entry_copy = entry.copy()
                entry_copy["mapped_vol"] = None
                updated_entries.append(entry_copy)
                continue
            
            # Process new poses since last entry
            if i == 0:
                # First entry - process all poses in this entry
                for rid, pose_list in robot_poses.items():
                    pose = np.array(pose_list, dtype=np.float64)
                    depth = self.capture_depth_at_pose(pose)
                    self.mapper.insert_depth_to_buffer(depth=depth, transform=pose)
            else:
                # Subsequent entries - process only new poses
                prev_poses = entries[i - 1].get("robot_poses", {})
                for rid, pose_list in robot_poses.items():
                    if rid not in prev_poses:
                        pose = np.array(pose_list, dtype=np.float64)
                        depth = self.capture_depth_at_pose(pose)
                        self.mapper.insert_depth_to_buffer(depth=depth, transform=pose)
            
            # Integrate depth buffer
            self.mapper.integrate_from_buffer()
            
            # Compute volume
            volume = self.compute_mapped_volume()
            
            logging.info(f"Step {i + 1}/{len(entries)}: mapped_vol = {volume:.4f} m^3")
            
            # Add volume to entry
            entry_copy = entry.copy()
            entry_copy["mapped_vol"] = volume
            updated_entries.append(entry_copy)
            
            # Save intermediate results every N steps
            if save_interval > 0 and (i + 1) % save_interval == 0:
                logging.info(f"Saving intermediate results at step {i + 1}")
                self.save_results(updated_entries, self.output_path)
        
        return updated_entries

    def save_results(self, entries: List[Dict[str, Any]], output_path: str) -> None:
        """Save updated entries to JSON file."""
        with open(output_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        
        logging.info(f"Results saved to {output_path}")

    def run(self) -> None:
        """Main replay loop."""
        # Setup
        self.setup_viewer()
        self.setup_mapper()
        
        # Load trajectory
        entries = self.load_trajectory()
        
        if len(entries) == 0:
            logging.error("No entries found in JSON file")
            return
        
        # Set output path
        self.output_path = self.args.output or self.args.json_file.replace(".json", "_with_volume.json")
        
        # Replay and compute volumes
        updated_entries = self.replay_trajectory(entries)
        
        # Save final results
        self.save_results(updated_entries, self.output_path)
        
        # Print summary
        volumes = [e.get("mapped_vol") for e in updated_entries if e.get("mapped_vol") is not None]
        if volumes:
            logging.info(f"Summary: {len(volumes)} steps processed")
            logging.info(f"  Final volume: {volumes[-1]:.4f} m^3")
            logging.info(f"  Max volume: {max(volumes):.4f} m^3")
        
        # Cleanup
        if self.vis:
            self.vis.destroy_window()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser matching demo_exploration.py style."""
    p = argparse.ArgumentParser(
        description="Replay exploration trajectory and compute mapped volume"
    )
    
    p.add_argument(
        "--mesh", 
        type=str, 
        required=True, 
        help="Path to the scene mesh file"
    )
    p.add_argument(
        "--json_file", 
        "-j",
        type=str, 
        required=True, 
        help="Path to the JSON file from demo_exploration"
    )
    p.add_argument(
        "--config",
        type=str,
        default="config/hm3d_exploration.yaml",
        help="Configuration YAML file (same as demo_exploration)"
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: input_with_volume.json)"
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum number of steps to replay"
    )
    p.add_argument(
        "--save_interval",
        "-s",
        type=int,
        default=10,
        help="Save results every N steps (0 to disable intermediate saves)"
    )
    p.add_argument(
        "--log_level",
        "-ll",
        type=int,
        default=20,
        help="Logging level (10=debug, 20=info, 30=warning)"
    )
    
    return p


def main():
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    
    print(f"Open3D version: {o3d.__version__}")
    
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.log_level < 20:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.log_level < 30:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    app = ReplayApp(args)
    app.run()


if __name__ == "__main__":
    main()