"""
Headless exploration demo using Open3D's OffscreenRenderer.
This version does not require a display and can run on clusters without monitors.
"""

import os
import time
import logging
import argparse
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import open3d.visualization.rendering as rendering

from utils.vis_utils import (
    create_camera,
)

# FrontierNet
from frontier.detector import FrontierDetector
from frontier.model.predict import load_model
from utils.frontier_utils import read_config_yaml

# Mapping
from mapping.wavemap import WaveMapper

# Frontier Manager
from frontier.manager import FrontierManager

# mono depth
from mono_depth.Metric3D import metric_depth_from_rgb as metric_depth_from_rgb_metric3d
from mono_depth.UniK3D import metric_depth_from_rgb as metric_depth_from_rgb_unik3d


class HeadlessRenderer:
    """
    A headless renderer using Open3D's OffscreenRenderer for rendering RGBD
    from a mesh without requiring a display.
    """

    def __init__(
        self,
        mesh_path: str,
        width: int,
        height: int,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        z_near: float = 0.02,
        z_far: float = 50.0,
    ):
        """
        Initialize the headless renderer.

        Args:
            mesh_path: Path to the mesh file.
            width: Image width in pixels.
            height: Image height in pixels.
            intrinsic: Camera intrinsic parameters.
            z_near: Near clipping plane.
            z_far: Far clipping plane.
        """
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.z_near = z_near
        self.z_far = z_far

        # Current camera extrinsic (C_T_W: camera frame expressed in world)
        self._extrinsic = np.eye(4)

        # Create the offscreen renderer
        self.renderer = rendering.OffscreenRenderer(width, height)

        # Load the mesh with materials using read_triangle_model for GLB/GLTF
        # This preserves textures and materials
        mesh_path_lower = mesh_path.lower()
        if mesh_path_lower.endswith('.glb') or mesh_path_lower.endswith('.gltf'):
            # Use read_triangle_model which preserves materials/textures
            model = o3d.io.read_triangle_model(mesh_path)
            if model is not None and len(model.meshes) > 0:
                # Add each mesh with its material
                for i, mesh_info in enumerate(model.meshes):
                    mesh_geom = mesh_info.mesh
                    material_idx = mesh_info.material_idx
                    
                    if material_idx >= 0 and material_idx < len(model.materials):
                        mat = model.materials[material_idx]
                        # Use "unlitLine" shader which shows pure colors without any lighting
                        # Or create a new unlit material with just the albedo texture
                        new_mat = rendering.MaterialRecord()
                        new_mat.shader = "defaultUnlit"
                        # Copy albedo image if exists
                        if mat.albedo_img is not None:
                            new_mat.albedo_img = mat.albedo_img
                        # Set base color to white so texture shows at full brightness
                        new_mat.base_color = [1.0, 1.0, 1.0, 1.0]
                        self.renderer.scene.add_geometry(f"mesh_{i}", mesh_geom, new_mat)
                    else:
                        # Fallback material
                        mat = rendering.MaterialRecord()
                        mat.shader = "defaultUnlit"
                        mat.base_color = [0.8, 0.8, 0.8, 1.0]
                        self.renderer.scene.add_geometry(f"mesh_{i}", mesh_geom, mat)
                print(f"Loaded GLB model with {len(model.meshes)} meshes and {len(model.materials)} materials")
            else:
                raise ValueError(f"Failed to load model from {mesh_path}")
        else:
            # For other formats, load as triangle mesh
            mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
            material = rendering.MaterialRecord()
            
            if mesh.has_vertex_colors():
                material.shader = "defaultUnlit"
            else:
                material.shader = "defaultLit"
                material.base_color = [0.8, 0.8, 0.8, 1.0]
            
            self.renderer.scene.add_geometry("mesh", mesh, material)
            print(f"Loaded mesh: {mesh}")
        
        # Disable lighting since we use unlit shader - show original colors
        self.renderer.scene.scene.enable_sun_light(False)
        self.renderer.scene.scene.enable_indirect_light(False)
        
        # Set background to white for better visibility
        self.renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

        # Setup initial camera
        self._setup_camera()

    def cleanup(self):
        """Clean up renderer resources."""
        try:
            self.renderer.scene.clear_geometry()
        except Exception:
            pass

    def _setup_camera(self):
        """Setup camera with current intrinsic and extrinsic."""
        # Get intrinsic matrix
        K = self.intrinsic.intrinsic_matrix

        # Setup camera using intrinsic matrix and extrinsic
        self.renderer.setup_camera(
            K,
            self._extrinsic,
            self.intrinsic.width,
            self.intrinsic.height,
        )

    def set_extrinsic(self, extrinsic: np.ndarray):
        """
        Set the camera extrinsic matrix.

        Args:
            extrinsic: 4x4 camera-to-world transformation matrix (C_T_W).
        """
        self._extrinsic = extrinsic.copy()
        self._setup_camera()

    def get_extrinsic(self) -> np.ndarray:
        """Get the current camera extrinsic matrix (C_T_W)."""
        return self._extrinsic.copy()

    def get_intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        """Get the camera intrinsic parameters."""
        return self.intrinsic

    def capture_rgb(self) -> np.ndarray:
        """
        Capture RGB image from the current viewpoint.

        Returns:
            RGB image as numpy array (H, W, 3) with values in [0, 255].
        """
        img = self.renderer.render_to_image()
        return np.asarray(img)

    def capture_depth(self) -> np.ndarray:
        """
        Capture depth image from the current viewpoint.

        Returns:
            Depth image as numpy array (H, W) with depth in meters.
        """
        # z_in_view_space=True gives us actual depth values (distance from camera)
        depth_img = self.renderer.render_to_depth_image(z_in_view_space=True)
        depth = np.asarray(depth_img).astype(np.float32)
        
        # Replace inf values (background/sky) with 0
        depth[~np.isfinite(depth)] = 0.0
        
        return depth

    def capture_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture both RGB and depth images.

        Returns:
            Tuple of (rgb, depth) numpy arrays.
        """
        rgb = self.capture_rgb()
        depth = self.capture_depth()
        return rgb, depth


class HeadlessExplorerApp:
    """
    A headless version of the exploration app that uses OffscreenRenderer
    instead of interactive visualization windows.
    """

    # ---------- constants / defaults ----------
    VOX_SIZE = 0.1

    # Camera (robot) defaults
    CAM_H, CAM_W, CAM_F = 480, 480, 300.0

    # Depth sources
    DEPTH_GT = "GT"
    DEPTH_M3D = "Metric3D"
    DEPTH_UNIK3D = "UniK3D"

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Config
        self.config = read_config_yaml(args.config)
        self.predict_interval: int = int(self.config.get("predict_interval", 5))
        self.plan_interval: int = int(self.config.get("plan_interval", 10))

        # Depth source
        self.depth_source = args.depth_source

        # Headless renderer
        self.renderer: Optional[HeadlessRenderer] = None

        # Frontier, mapping, detector
        self.mapper: Optional[WaveMapper] = None
        self.ft_manager: Optional[FrontierManager] = None
        self.ft_detector: Optional[FrontierDetector] = None
        self.VOX_SIZE = (
            self.config["voxel_size"]
            if self.config["voxel_size"] is not None
            else self.VOX_SIZE
        )

        # Path & motion tracking
        self.path_to_go: List[np.ndarray] = []
        self.move_enough: bool = True
        self.last_W_T_C: np.ndarray = np.eye(4)  # camera pose

        # JSON output
        save_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(save_dir, exist_ok=True)
        self.json_path: str = args.write_path or None

    # ---------- RGBD capture ----------

    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture RGB and depth using the configured depth source.

        Returns:
            Tuple of (rgb, depth) numpy arrays.
        """
        assert self.renderer is not None

        rgb = self.renderer.capture_rgb()
        K = self.renderer.get_intrinsic().intrinsic_matrix

        if self.depth_source == self.DEPTH_GT:
            depth = self.renderer.capture_depth()
        elif self.depth_source == self.DEPTH_UNIK3D:
            depth = metric_depth_from_rgb_unik3d(rgb_input=rgb, intrinsic_mat=K)
        elif self.depth_source == self.DEPTH_M3D:
            depth = metric_depth_from_rgb_metric3d(
                rgb_input=rgb,
                intrinsic_mat=K,
                camera_W=rgb.shape[1],
                camera_H=rgb.shape[0],
                local_model_path=None,
            )
        else:
            raise ValueError(f"Unknown depth_source: {self.depth_source}")

        return rgb, depth

    def is_moving(self, current_pose: np.ndarray, trans_thre: float = 0.1, rot_thre: float = 0.26) -> bool:
        """Check if the robot has moved enough from the last recorded pose."""
        from utils.geometry import pose_difference

        trans_diff, rot_diff = pose_difference(
            self.last_W_T_C.reshape(1, 4, 4), current_pose.reshape(1, 4, 4)
        )
        return trans_diff[0, 0] > trans_thre or rot_diff[0, 0] > rot_thre

    # ---------- main exploration logic ----------

    def exploration(self) -> None:
        """
        Main exploration loop (headless version).
        """
        assert self.renderer is not None
        assert self.ft_manager is not None and self.mapper is not None

        max_steps = self.args.max_steps
        max_time_s = self.args.max_time
        start_time = time.time()

        # Initial mapping (one frame) to bootstrap map
        _, depth0 = self.get_rgbd()
        C_T_W = self.renderer.get_extrinsic()
        W_T_C = np.linalg.inv(C_T_W)
        self.mapper.insert_depth_to_buffer(depth=depth0, transform=W_T_C)
        logging.info("Initial mapping round started.")
        self.mapper.integrate_from_buffer()
        self.mapper.interpolate_occupancy_grid()
        og = self.mapper.get_occupancy_grid()
        self.ft_manager.update_map(free_map=og["free"], occ_map=og["occupied"])

        while True:
            C_T_W = self.renderer.get_extrinsic()
            W_T_C = np.linalg.inv(C_T_W)
            n_robot_poses = len(self.ft_manager.robot_poses)

            logging.info(" -------Current exploration step: %d -------", n_robot_poses)

            if n_robot_poses > max_steps:
                logging.info("Maximum steps reached, exploration finished.")
                break

            if time.time() - start_time > max_time_s:
                logging.info("Time limit reached, exploration finished.")
                break

            no_more_frontier = (
                len(self.ft_manager.valid_frontiers) == 0 and n_robot_poses > 10
            )
            reach_next_update = len(self.path_to_go) == 0 or (
                (n_robot_poses - 1) % self.predict_interval == 0
            )

            if no_more_frontier or reach_next_update:
                logging.info("Updating frontiers.")
                # New observation
                rgb, depth = self.get_rgbd()

                # Frontier detection + anchoring
                self.ft_detector.detect(
                    rgb=rgb,
                    depth=depth,
                    df_normalizer=self.config["df_normalizer"],
                    df_thr=self.config["df_thr"],
                )
                ft_list = self.ft_detector.anchor_fts(depth=depth, extrinsic=C_T_W)

                # Add into manager
                if ft_list:
                    new_ids = self.ft_manager.add_robot_poses([W_T_C])
                    self.ft_manager.add_frontiers(frontiers=ft_list, parent_ids=new_ids)
                    self.ft_manager.filter_frontiers()
                    self.ft_manager.gain_adjustment()
                    self.ft_manager.filter_frontiers()

                if len(self.ft_manager.valid_frontiers) == 0:
                    logging.info("No frontiers, exploration finished.")
                    break

            # Update mapper continuously
            self.mapper.integrate_from_buffer()
            self.mapper.interpolate_occupancy_grid()

            og = self.mapper.get_occupancy_grid()
            self.ft_manager.update_map(free_map=og["free"], occ_map=og["occupied"])
            self.ft_manager.gain_adjustment()
            self.ft_manager.filter_frontiers()
            self.ft_manager.merge_frontiers()
            self.ft_manager.filter_frontiers()
            self.ft_manager.update_utility(current_pos=W_T_C[:3, 3])

            # Replan if needed
            if reach_next_update and self.move_enough:
                logging.info("Replanning...")
                logging.debug(f"Replanning (interval={self.plan_interval}).")
                self.path_to_go = self.ft_manager.plan_path_to_goal(W_T_C) or []
                if self.path_to_go:
                    logging.info(
                        f"Path to goal found with {len(self.path_to_go)} steps."
                    )
                    self.move_enough = False
                else:
                    logging.warning("No path found, deleting current goal frontier.")
                    self.path_to_go = []
                    self.move_enough = True  # try again next cycle

            # Persist state snapshot
            if self.json_path:
                logging.info(f"Writing state to {self.json_path}")
                self.ft_manager.write_to_file(file_path=self.json_path)

            # Execute one movement step if path exists
            if self.path_to_go:
                logging.debug("Moving along the path.")
                self.move(steps=1)

        # Final state output
        logging.info("Exploration finished, total steps: %d", n_robot_poses)

    # ---------- motion & mapping ----------

    def move(self, steps: int) -> None:
        """
        Execute up to `steps` motions along the path, acquire depth, and update mapper & manager.
        """
        if self.renderer is None:
            return
        if not self.path_to_go:
            logging.info("No path to follow.")
            return

        for _ in range(steps):
            if not self.path_to_go:
                logging.info("Path exhausted.")
                break

            next_W_T_C = self.path_to_go.pop(0)
            logging.debug(f"Moving to next pose:\n{next_W_T_C}")

            # Update renderer camera extrinsic (needs C_T_W)
            self.renderer.set_extrinsic(np.linalg.inv(next_W_T_C))

            # Capture new depth
            _, depth = self.get_rgbd()

            # Insert into mapper
            C_T_W = self.renderer.get_extrinsic()
            W_T_C = np.linalg.inv(C_T_W)
            self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C)

            # Check if we truly moved
            if self.is_moving(W_T_C):
                self.last_W_T_C = W_T_C
                if self.ft_manager is not None:
                    self.ft_manager.add_robot_poses([W_T_C])
                self.move_enough = True

    # ---------- setup ----------

    def setup_system(self) -> None:
        """Initialize renderer, mapper, detector, and manager."""
        # Configure logging level
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

        # Load scene mesh (just for logging, HeadlessRenderer will load it properly)
        logging.info(f"Loading mesh from: {self.args.mesh}")

        # Create camera intrinsics
        cam_intrinsic = create_camera(self.CAM_H, self.CAM_W, self.CAM_F)

        # Create headless renderer - pass mesh path for proper texture loading
        self.renderer = HeadlessRenderer(
            mesh_path=self.args.mesh,
            width=self.CAM_W,
            height=self.CAM_H,
            intrinsic=cam_intrinsic,
            z_near=0.02,
            z_far=50.0,
        )

        # Set initial camera pose from config
        initial_C_T_W = np.asarray(self.config["initial_cam_extrinsic"], dtype=float)
        self.renderer.set_extrinsic(initial_C_T_W)
        self.last_W_T_C = np.linalg.inv(initial_C_T_W)

        # Mapper
        intr = cam_intrinsic
        params = {
            "min_cell_width": self.VOX_SIZE / 2.0,
            "width": intr.width,
            "height": intr.height,
            "fx": intr.intrinsic_matrix[0, 0],
            "fy": intr.intrinsic_matrix[1, 1],
            "cx": intr.intrinsic_matrix[0, 2],
            "cy": intr.intrinsic_matrix[1, 2],
            "min_range": 0.05,
            "max_range": (
                self.config["depth_range"]
                if self.config["depth_range"] is not None
                else 3.5
            ),
            "resolution": self.VOX_SIZE,
        }
        self.mapper = WaveMapper(params=params)

        # FrontierNet
        unet = load_model(
            path=self.args.unet_weight,
            num_classes=self.config["num_classes"],
            use_depth=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ft_detector = FrontierDetector(
            model=unet,
            camera_intrinsic=intr.intrinsic_matrix.copy(),
            use_depth=True,
            img_size_model=self.config["input_img_size"],
            device=device,
            log_level=self.args.log_level,
        )

        # Frontier Manager
        self.ft_manager = FrontierManager(
            params=self.config, log_level=self.args.log_level
        )

        logging.info("Headless system setup complete.")

    def cleanup(self) -> None:
        """Clean up resources to avoid memory issues on exit."""
        # Explicitly delete renderer to clean up GPU resources before exit
        if self.renderer is not None:
            self.renderer.cleanup()
            del self.renderer.renderer
            del self.renderer
            self.renderer = None
        # Force garbage collection
        import gc
        gc.collect()

    def run(self) -> None:
        """Run the headless exploration."""
        try:
            self.setup_system()
            logging.info("Starting headless exploration...")
            self.exploration()
            logging.info("Headless exploration complete.")
        finally:
            self.cleanup()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Headless FrontierNet exploration demo (no display required)"
    )
    p.add_argument("--mesh", type=str, required=True, help="Path to the mesh file")
    p.add_argument(
        "--config",
        type=str,
        default="config/hm3d_exploration.yaml",
        help="FrontierNet configuration file",
    )
    p.add_argument(
        "--write_path", type=str, help="JSON file to write the ftmanager state"
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of exploration steps",
    )
    p.add_argument(
        "--max_time", type=int, default=3600, help="Maximum exploration time in seconds"
    )
    p.add_argument(
        "--unet_weight",
        type=Path,
        default=Path("model_weights/rgbd_11cls.pth"),
        help="Path to UNet model weights",
    )
    p.add_argument(
        "--depth_source",
        type=str,
        default=HeadlessExplorerApp.DEPTH_GT,
        choices=[
            HeadlessExplorerApp.DEPTH_GT,
            HeadlessExplorerApp.DEPTH_M3D,
            HeadlessExplorerApp.DEPTH_UNIK3D,
        ],
        help="Depth source",
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

    app = HeadlessExplorerApp(args)
    app.run()


if __name__ == "__main__":
    main()
