import os
import time
import threading
import logging
import argparse
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np
import torch
import open3d as o3d

from utils.vis_utils import (
    get_vis_state,
    set_vis_cam_ex,
    set_vis_cam_intr,
    is_vis_moving,
    camera_vis_with_cylinders,
    capture_rgb,
    capture_depth,
    create_camera,
    create_interactive_vis,
    load_mesh,
    register_basic_callbacks,
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


class ExplorerApp:
    """
    A lightweight wrapper for the exploration loop and state.
    """

    # ---------- constants / defaults ----------
    REFRESH_RATE = 50  # Hz
    VOX_SIZE = 0.1

    # Camera-1 (observer) defaults
    CAM1_H, CAM1_W, CAM1_F = 960, 1280, 700.0
    # Camera-2 (robot) defaults
    CAM2_H, CAM2_W, CAM2_F = 480, 480, 300.0

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

        # Visualization windows
        self.vis_1 = None  # observer view
        self.vis_2 = None  # robot view

        # Sensors
        ## set custom camera parameters if needed
        # self.CAM2_H = args.image_height if args.image_height is not None else self.CAM2_H
        # self.CAM2_W = args.image_width if args.image_width is not None else self.CAM2_W
        # self.CAM2_F = args.focal_length if args.focal_length is not None else self.CAM2_F

        # Frontier, mapping, detector
        self.mapper: Optional[WaveMapper] = None
        self.ft_manager: Optional[FrontierManager] = None
        self.ft_detector: Optional[FrontierDetector] = None
        self.VOX_SIZE = (
            self.config["voxel_size"]
            if self.config["voxel_size"] is not None
            else self.VOX_SIZE
        )

        # Geometry caches (for visualization in o3d)
        self.geometry_vis_1: List[o3d.geometry.Geometry] = []
        self.ft_geometry_vis_1: List[o3d.geometry.Geometry] = []

        # Path & motion tracking
        self.path_to_go: List[np.ndarray] = []
        self.move_enough: bool = True
        self.last_W_T_C2: np.ndarray = np.eye(4)  # camera pose in vis_2

        # JSON output
        save_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(save_dir, exist_ok=True)
        self.json_path: str = args.write_path or None

        # mutex for vis updates
        self._lock = threading.Lock()

    # ---------- visualization helpers ----------

    def update_vis(self) -> None:
        """Update both viewers while preserving their camera extrinsics."""
        if self.vis_1 is None or self.vis_2 is None:
            return

        cam_ex_1 = get_vis_state(self.vis_1)["cam_extrinsic"]
        cam_ex_2 = get_vis_state(self.vis_2)["cam_extrinsic"]

        # If vis_2 moved enough, refresh frustum and overlays
        if is_vis_moving(self.vis_2, self.last_W_T_C2, trans_thre=0.1, rot_thre=0.26):
            # Store latest
            self.last_W_T_C2 = np.linalg.inv(cam_ex_2)
            self.update_geometry_vis_1()

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

        # vis_2 camera poses
        C2_T_W = get_vis_state(self.vis_2)["cam_extrinsic"]
        W_T_C2 = np.linalg.inv(C2_T_W)

        intr = get_vis_state(self.vis_2)["cam_intrinsic"]
        W2, H2 = intr.width, intr.height
        fx2 = intr.intrinsic_matrix[0, 0]
        # fy2 = intr.intrinsic_matrix[1, 1]

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

        # Frontiers overlay
        self.visualize_all_frontiers(self.vis_1)

        # Frustum overlay
        for g in frustum_meshes:
            self.geometry_vis_1.append(g)
            self.vis_1.add_geometry(g, reset_bounding_box=False)

        if self.args.vis_graph:
            # (optional) topo-graph overlay
            graph_vis = self.ft_manager.get_graph_vis()
            for g in graph_vis:
                self.geometry_vis_1.append(g)
                self.vis_1.add_geometry(g, reset_bounding_box=False)

    def get_rgbd(
        self, vis, return_rgb_type: str = "np", return_depth_type: str = "np"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture RGB and depth for a given viewer using the configured depth source.
        """
        rgb = capture_rgb(vis, return_rgb_type=return_rgb_type)
        img = rgb["image"]
        if np.max(img) <= 1:
            img = (img * 255).astype(np.uint8)

        K = get_vis_state(vis)["cam_intrinsic"].intrinsic_matrix

        if self.depth_source == self.DEPTH_GT:
            depth = capture_depth(vis, return_depth_type=return_depth_type)["depth"]
        elif self.depth_source == self.DEPTH_UNIK3D:
            depth = metric_depth_from_rgb_unik3d(rgb_input=img, intrinsic_mat=K)
        elif self.depth_source == self.DEPTH_M3D:
            depth = metric_depth_from_rgb_metric3d(
                rgb_input=img,
                intrinsic_mat=K,
                camera_W=img.shape[1],
                camera_H=img.shape[0],
                local_model_path=None,
            )
        else:
            raise ValueError(f"Unknown depth_source: {self.depth_source}")

        return img, depth

    # ---------- main exploration logic ----------

    def exploration(self) -> None:
        """
        Main exploration loop.
        """
        assert self.vis_1 is not None and self.vis_2 is not None
        assert self.ft_manager is not None and self.mapper is not None

        self.vis_1.update_renderer()
        self.vis_2.update_renderer()

        max_steps = self.args.max_steps
        max_time_s = self.args.max_time
        start_time = time.time()

        # Initial mapping (one frame) to bootstrap map
        _, depth0 = self.get_rgbd(self.vis_2)
        C2_T_W = get_vis_state(self.vis_2)["cam_extrinsic"]
        W_T_C2 = np.linalg.inv(C2_T_W)
        self.mapper.insert_depth_to_buffer(
            depth=depth0, transform=W_T_C2
        )  # camera IN world frame
        logging.info("Initial mapping round started.")
        self.mapper.integrate_from_buffer()
        self.mapper.interpolate_occupancy_grid()
        og = self.mapper.get_occupancy_grid()
        self.ft_manager.update_map(free_map=og["free"], occ_map=og["occupied"])

        while True:

            C2_T_W = get_vis_state(self.vis_2)["cam_extrinsic"]
            W_T_C2 = np.linalg.inv(C2_T_W)
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
                rgb, depth = self.get_rgbd(self.vis_2)

                # Frontier detection + anchoring
                self.ft_detector.detect(
                    rgb=rgb,
                    depth=depth,
                    df_normalizer=self.config["df_normalizer"],
                    df_thr=self.config["df_thr"],
                )
                ft_list = self.ft_detector.anchor_fts(depth=depth, extrinsic=C2_T_W)

                # Add into manager
                if ft_list:
                    new_ids = self.ft_manager.add_robot_poses([W_T_C2])
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
            self.ft_manager.update_utility(current_pos=W_T_C2[:3, 3])

            # Replan if needed
            if reach_next_update and self.move_enough:
                logging.info("Replanning...")
                logging.debug(f"Replanning (interval={self.plan_interval}).")
                self.path_to_go = self.ft_manager.plan_path_to_goal(W_T_C2) or []
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

        # Cleanly close
        logging.info("Exploration finished, total steps: %d", n_robot_poses)

    # ---------- visualization: frontiers ----------

    def visualize_all_frontiers(self, vis) -> None:
        """
        Overlay all valid frontiers in vis_1 (frustums + axes + goal marker).
        """
        if self.ft_manager is None:
            return

        C_T_W = get_vis_state(vis)["cam_extrinsic"]  # stash/restore
        for g in self.ft_geometry_vis_1:
            vis.remove_geometry(g)
        self.ft_geometry_vis_1.clear()

        if len(self.ft_manager.all_frontiers) == 0:
            logging.debug("No frontiers to visualize.")
            set_vis_cam_ex(vis, C_T_W)
            return

        # Draw each frontier frustum
        for ft in self.ft_manager.valid_frontiers:
            W_T_C = ft.pose6d  # already W_T_C
            frustum = camera_vis_with_cylinders(
                W_T_C,
                wh_ratio=self.CAM2_W / self.CAM2_H,
                scale=0.6,
                weight=(ft.u_gain or 0.0) / 20.0,
                fovx=2 * np.degrees(np.arctan(self.CAM2_W / (2 * self.CAM2_F))),
                radius=0.04,
                return_mesh=False,
            )
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0]
            )
            axis.transform(W_T_C)
            frustum.append(axis)
            for g in frustum:
                self.ft_geometry_vis_1.append(g)
                vis.add_geometry(g, reset_bounding_box=False)

        # Mark current goal
        try:
            goal_pose = self.ft_manager.goal_pose
            if goal_pose is not None:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
                sphere.paint_uniform_color([0, 0, 1])
                sphere.translate(goal_pose[:3, 3])
                vis.add_geometry(sphere, reset_bounding_box=False)
                self.ft_geometry_vis_1.append(sphere)
        except Exception as e:
            logging.error(f"Error visualizing next goal frontier: {e}")

        set_vis_cam_ex(vis, C_T_W)

    # ---------- motion & mapping ----------

    def move(self, steps: int) -> None:
        """
        Execute up to `steps` motions along the path, acquire depth, and update mapper & manager.
        """
        if self.vis_1 is None or self.vis_2 is None:
            return
        if not self.path_to_go:
            logging.info("No path to follow.")
            return

        C1_T_W = get_vis_state(self.vis_1)["cam_extrinsic"]

        for _ in range(steps):
            if not self.path_to_go:
                logging.info("Path exhausted.")
                break

            next_W_T_C = self.path_to_go.pop(0)
            logging.debug(f"Moving to next pose:\n{next_W_T_C}")

            # Update vis_2 camera extrinsic
            set_vis_cam_ex(self.vis_2, np.linalg.inv(next_W_T_C))

            # Capture new depth
            _, depth = self.get_rgbd(self.vis_2)

            # Insert into mapper
            C2_T_W = get_vis_state(self.vis_2)["cam_extrinsic"]
            W_T_C2 = np.linalg.inv(C2_T_W)
            self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C2)

            # Update viewers
            self.vis_1.update_renderer()
            self.vis_2.update_renderer()
            self.vis_1.poll_events()
            self.vis_2.poll_events()
            time.sleep(0.1)

            # If we truly moved, update path bookkeeping
            if is_vis_moving(
                self.vis_2, self.last_W_T_C2, trans_thre=0.1, rot_thre=0.26
            ):
                self.last_W_T_C2 = W_T_C2
                if self.ft_manager is not None:
                    self.ft_manager.add_robot_poses([W_T_C2])
                self.move_enough = True

        # Refresh overlays
        self.update_geometry_vis_1()
        self.vis_1.update_renderer()
        self.vis_2.update_renderer()
        set_vis_cam_ex(self.vis_1, C1_T_W)

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
        self.vis_1.add_geometry(scene_mesh, reset_bounding_box=True)
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

    def setup_system(self) -> None:
        """Init mapper, detector, and manager."""
        # Mapper
        intr2 = get_vis_state(self.vis_2)["cam_intrinsic"]
        params = {
            "min_cell_width": self.VOX_SIZE / 2.0,
            "width": intr2.width,
            "height": intr2.height,
            "fx": intr2.intrinsic_matrix[0, 0],
            "fy": intr2.intrinsic_matrix[1, 1],
            "cx": intr2.intrinsic_matrix[0, 2],
            "cy": intr2.intrinsic_matrix[1, 2],
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
            camera_intrinsic=intr2.intrinsic_matrix.copy(),
            use_depth=True,
            img_size_model=self.config["input_img_size"],
            device=device,
            log_level=self.args.log_level,
        )

        # Frontier Manager
        self.ft_manager = FrontierManager(
            params=self.config, log_level=self.args.log_level
        )

    def run(self) -> None:
        self.setup_viewers()
        self.setup_system()

        if self.args.auto_start:
            self.exploration()

        else:
            logging.info(
                " --- You are in manual mode, you can move the camera using WASD(translation), JL(rotation), QZ(height) keys in the small window ---"
            )

            def on_space(vis):
                self.exploration()
                return False

            self.vis_1.register_key_callback(32, on_space)
            self.vis_2.register_key_callback(32, on_space)
            logging.info(" --- PRESS SPACE TO START EXPLORATION --- ")

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
    p = argparse.ArgumentParser()
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
        "--auto_start",
        action="store_true",
        default=False,
        help="Auto-start exploration loop",
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
        "--vis_graph",
        action="store_true",
        default=False,
        help="Visualize the topological graph",
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
        default=ExplorerApp.DEPTH_GT,
        choices=[ExplorerApp.DEPTH_GT, ExplorerApp.DEPTH_M3D, ExplorerApp.DEPTH_UNIK3D],
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

    app = ExplorerApp(args)
    app.run()


if __name__ == "__main__":
    main()
