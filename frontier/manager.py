import json
from typing import Optional, Iterable, List, Dict
import logging
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from frontier.base import Base
from frontier.frontier import Frontier
from frontier.graph import FrontierGraph

from utils.frontier_utils import ft_pos_direct_distance
from utils.geometry import compute_alignment_transforms, pose_difference
from utils.vis_utils import create_cylinder_between_points
from utils.mapping_utils import (
    render_voxel_depth,
    select_visible_points,
    compute_visible_voxels,
)
from planner.occ_rrt_point3d import OccupancyGrid3DPathPlanner as PathPlanner


class FrontierManager(Base):

    def __init__(self, params: Optional[dict] = None, log_level: int = logging.INFO) -> None:
            """
            params keys (optional):
            - filter_bbox: list[6] or None
            - filter_max_vd_z: float or None
            - filter_min_gain: float
            - visited_trans_threshold: float
            - visited_angle_threshold: float
            - visited_gain_reduction_factor: float
            - render_K: (3,3) intrinsics
            - render_H: int
            - render_W: int
            - render_depth_range: float
            - voxel_size: float
            - render_decrease_factor: float
            - force_all_frontier_to_xy: bool
            - utility_gain_factor: float
            - max_planning_time: float
            - planning_algo: str
            - (some planner-related keys are forwarded to PathPlanner)
            """
            super().__init__(params, log_level=log_level)
            self.logger.info("FrontierManager initialized")

            p = params or {}

            # frontiers and robot poses
            self.frontiers: Dict[int, Frontier] = {}
            self.robot_poses: Dict[int, np.ndarray] = {}

            # Graph/Maps
            self.graph = FrontierGraph(p, log_level=log_level)
            self.occ_map: Optional[np.ndarray] = None
            self.free_map: Optional[np.ndarray] = None

            # Planner
            self.planner = PathPlanner(p)
            self.current_goal_pose: Optional[np.ndarray] = None
            self.current_goal_ft_id: Optional[int] = None

            # IDs
            self._current_frontier_id: int = self.graph.type_range["F"][0]
            self._current_robot_id: int = self.graph.type_range["R"][0]

            # Parameters
            self.filter_bbox = p.get("filter_bbox", None)
            self.filter_max_vd_z = p.get("filter_max_vd_z", None)
            self.filter_min_gain = float(p.get("filter_min_gain", 0.1))
            self.max_planning_time = float(p.get("max_planning_time", 1.5))
            self.planning_algo = p.get("planning_algo", "rrtstar")

            self.v_tras_thre = float(p.get("visited_trans_threshold", 0.4))
            self.v_angl_thre = float(p.get("visited_angle_threshold", 0.4))
            self.v_gain_reduction_factor = float(p.get("visited_gain_reduction_factor", 1000))

            # Rendering
            if p.get("render_K") is not None:
                self.render_K = np.array(p["render_K"], dtype=np.float32).reshape(3, 3)
            else:
                self.render_K = np.array([[60, 0, 64], [0, 60, 64], [0, 0, 1]], dtype=np.float32)

            self.render_H = int(p.get("render_H", 128))
            self.render_W = int(p.get("render_W", 128))
            self.render_d_range = float(p.get("render_depth_range", 3.5))
            self.voxel_size = float(p.get("voxel_size", 0.1))
            self.render_decrease_factor = float(p.get("render_decrease_factor", 1.0))

            self.force_all_frontier_to_xy = bool(p.get("force_all_frontier_to_xy", False))
            self.utility_g_factor = float(p.get("utility_gain_factor", 1.0))

            # Validation
            if self.filter_bbox is not None:
                if not (isinstance(self.filter_bbox, (list, tuple)) and len(self.filter_bbox) == 6):
                    raise ValueError("filter_bbox must be a list/tuple of 6 numbers [xmin,xmax,ymin,ymax,zmin,zmax].")
            if self.voxel_size <= 0:
                raise ValueError("voxel_size must be positive.")
            if self.render_H <= 0 or self.render_W <= 0:
                raise ValueError("render_H and render_W must be positive.")
            if self.render_d_range <= 0:
                raise ValueError("render_depth_range must be positive.")

    def _alloc_frontier_id(self) -> int:
        """
        Return a fresh frontier ID. Skips over any IDs that might
        already be in use (e.g., after deletions).
        """
        nid = self._current_frontier_id
        # Advance counter for next time
        self._current_frontier_id += 1
        # If this nid is somehow in use, keep advancing
        while nid in self.frontiers:
            nid = self._current_frontier_id
            self._current_frontier_id += 1
        return nid


    def _alloc_robot_id(self) -> int:
        """
        Return a fresh robot ID. Skips over any IDs that might
        already be in use (e.g., after deletions).
        """
        rid = self._current_robot_id
        # Advance counter for next time
        self._current_robot_id += 1
        # If this rid is somehow in use, keep advancing
        while rid in self.robot_poses:
            rid = self._current_robot_id
            self._current_robot_id += 1
        return rid

    @property
    def all_frontiers(self) -> List[Frontier]:
        return list(self.frontiers.values())

    @property
    def all_frontiers_ids(self) -> List[int]:
        return list(self.frontiers.keys())

    @property
    def valid_frontiers(self) -> List[Frontier]:
        return [ft for ft in self.frontiers.values() if ft.is_valid]

    @property
    def utility(self) -> List[float]:
        return [ft.utility for ft in self.valid_frontiers]

    @property
    def current_robot_id(self) -> Optional[int]:
        """
        Latest existing robot ID (None if no robot poses yet).
        Previously this returned the *next* ID; this is safer.
        """
        if self.robot_poses:
            return max(self.robot_poses.keys())
        return None
    
    @property
    def goal_pose(self) -> np.ndarray | None:
        return self.current_goal_pose

    def add_robot_poses(self, robot_poses: list[np.ndarray]):
        """
        Add a list of robot poses to the manager.
        Args:
            robot_poses: List of robot poses, each pose is a 4x4 matrix.

        Returns:
            List of added robot IDs.
        """
        if len(robot_poses) == 0:
            self.logger.debug("No robot poses to add.")
            return []
        
        added_robot_ids = []
        for pose in robot_poses:
            assert pose.shape == (4, 4), "Each robot pose must be a 4x4 matrix."
            robot_id = self._alloc_robot_id()  # Assign a unique ID
            self.robot_poses[robot_id] = pose
            # add the robot node to the graph
            self.graph.add_node_R(robot_id)
            added_robot_ids.append(robot_id)
            # link the node with the latest robot pose (as the robot ID increments, the latest pose is always the last one added)
            if robot_id -1 in self.robot_poses: # avoid the first robot pose
                prev_pose = self.robot_poses[robot_id - 1]
                distance = np.linalg.norm(pose[:3, 3] - prev_pose[:3, 3])  # Euclidean distance in 3D space
                self.graph.add_edge_RR(robot_id, robot_id-1, weight=distance)  
        
        return added_robot_ids  # Return the list of added robot IDs
    
    def get_frontier(self, frontier_id: int, *, required: bool = False) -> Optional[Frontier]:
        """
        Fetch a frontier by ID.

        Args:
            frontier_id: Frontier ID to fetch.
            required: If True, raise KeyError when missing; otherwise return None.

        Returns:
            Frontier or None.
        """
        ft = self.frontiers.get(frontier_id)
        if ft is None:
            msg = f"Frontier ID {frontier_id} not found."
            if required:
                raise KeyError(msg)
            self.logger.debug(msg)
        return ft


    def get_robot_pose(self, robot_id: int, *, required: bool = False) -> Optional[np.ndarray]:
        """
        Fetch a robot pose (4x4) by ID.

        Args:
            robot_id: Robot ID to fetch.
            required: If True, raise KeyError when missing; otherwise return None.

        Returns:
            np.ndarray shape (4,4) or None.
        """
        pose = self.robot_poses.get(robot_id)
        if pose is None:
            msg = f"Robot pose ID {robot_id} not found."
            if required:
                raise KeyError(msg)
            self.logger.debug(msg)
        return pose

    @staticmethod
    def get_frontier_pose(frontier) -> np.ndarray:
        """
        Build a 4x4 pose for a frontier by aligning camera +Z to the frontier's
        view_direction and placing it at pos3d. Uses CV camera convention:
        +Z forward, +Y down (approx), +X right.

        Returns:
            (4,4) float ndarray
        """
        pos = np.asarray(frontier.pos3d, dtype=float).reshape(3)
        vd  = np.asarray(frontier.view_direction, dtype=float).reshape(3)
        
        T = compute_alignment_transforms(
            origins=[pos],
            align_vec=vd,
            align_axis=[0, 0, 1],
            appr_vec=[0, 0, -1],  # CV convention: Y down approx
            appr_axis=[0, 1, 0],
        )[0]

        # Ensure exact translation (in case the util modifies it)
        T = np.asarray(T, dtype=float)
        T[:3, 3] = pos
        return T
    
    def add_frontiers(self, frontiers: List[Frontier], parent_ids: Optional[Iterable[int]] = None) -> List[int]:
        """
        Add a list of frontiers. Optionally connect each to given robot parent IDs.

        Args:
            frontiers: Frontier objects to add.
            parent_ids: Robot pose IDs to connect (edges F-R). Duplicates ignored.
        """
        if len(frontiers) == 0:
            self.logger.debug("No frontiers to add.")
            return
        else:
            self.logger.debug(f"Adding {len(frontiers)} frontiers.")

        parent_ids = list(parent_ids) if parent_ids is not None else []

        for frontier in frontiers:
            # if force_all_frontier_to_xy is True, set the 3d vector to be in the XY plane
            if self.force_all_frontier_to_xy:
                vd = np.array([frontier.view_direction[0], frontier.view_direction[1], 0.0], dtype=float)
                n = np.linalg.norm(vd)
                frontier.view_direction = vd / n if n > 1e-8 else vd           
            frontier.id = self._alloc_frontier_id()  # Assign a unique ID
            self.frontiers[frontier.id] = frontier  # Use ID as key for fast access
            # add the frontier node to the graph
            self.graph.add_node_F(frontier.id)
            # set the frontier's parent IDs
            frontier.parent_ids = []
            # assign 6D pose
            frontier.pose6d = self.get_frontier_pose(frontier) 

            for parent_id in parent_ids:
                assert parent_id in self.robot_poses, f"Parent robot ID {parent_id} does not exist in robot poses."
                # calculate the distance between frontier position and the robot position
                distance = np.linalg.norm(frontier.pos3d - self.robot_poses[parent_id][:3, 3])
                self.graph.add_edge_FR(frontier.id, parent_id, weight= distance)
                frontier.parent_ids.append(parent_id)  # Add parent ID to the frontier
    
    def remove_frontiers(self, frontier_ids):
        """
        Remove frontiers by their IDs.
        Args:
            frontier_ids: List of frontier IDs to remove.
        """
        if not isinstance(frontier_ids, list):
            raise ValueError("frontier_ids must be a list.")
        
        # Deduplicate while preserving order
        seen = set()
        ids = [fid for fid in frontier_ids if not (fid in seen or seen.add(fid))]

        if not ids:
            self.logger.debug("No frontier IDs provided to remove.")
            return

        removed_any = False
        for fid in ids:
            if fid in self.frontiers:
                del self.frontiers[fid]
                try:
                    self.graph.remove_node_F(fid) 
                except Exception as e:
                    self.logger.warning(f"Graph removal failed for frontier {fid}: {e}")
                removed_any = True

        # Clear goal if it was removed
        if removed_any and self.current_goal_ft_id is not None and self.current_goal_ft_id not in self.frontiers:
            self.current_goal_ft_id = None
            self.current_goal_pose = None

        if removed_any:
            self.logger.debug(f"Removed frontiers: {ids}")
        else:
            self.logger.debug("No matching frontiers to remove.")

    def remove_invalid_frontiers(self) -> None:
        """
        Remove all frontiers currently marked invalid.
        """
        # Build a stable list before mutating self.frontiers
        invalid_ids = [fid for fid, ft in list(self.frontiers.items()) if not ft.is_valid]

        if not invalid_ids:
            self.logger.debug("No invalid frontiers to remove.")
            return

        self.remove_frontiers(invalid_ids)
        self.logger.debug(f"Removed {len(invalid_ids)} invalid frontiers.")
            

    def merge_frontiers(self):
        dbscan_params = {
            "eps": 1.8,
            "min_samples": 1,
            "weights": [1, 2],
        }

        valid_frontiers = list(self.valid_frontiers) 
        _n_valid_before = len(valid_frontiers)
        if _n_valid_before < 2:
            self.logger.debug("Not enough valid frontiers to merge.")
            return

        positions = np.array([ft.pos3d for ft in valid_frontiers])
        directions = np.array([ft.view_direction for ft in valid_frontiers])
        ft_features = np.concatenate((positions, directions), axis=1)

        metric_func = lambda x, y: ft_pos_direct_distance(x, y, weights=dbscan_params["weights"])
        dbscan = DBSCAN(
            eps=dbscan_params["eps"],
            min_samples=dbscan_params["min_samples"],
            metric=metric_func,
            n_jobs=-1,
        )
        labels = dbscan.fit_predict(ft_features)
        _n_valid_before = len(valid_frontiers)

        # Prepare merges
        to_remove = []
        to_add = []

        for cls_id in np.unique(labels):
            if cls_id == -1:
                continue
            _fts = [ft for i, ft in enumerate(valid_frontiers) if labels[i] == cls_id]
            if len(_fts) <= 1:
                continue
            else:
                _ft = Frontier()
                _gains = np.array([max(ft.u_gain, 1e-4) for ft in _fts])
                _ft.pos3d = np.average([ft.pos3d for ft in _fts], axis=0, weights=_gains)
                cls_vd = np.sum(np.array([ft.view_direction for ft in _fts]), axis=0)
                _ft.view_direction = cls_vd / np.linalg.norm(cls_vd) + 1e-6
                _ft.direct_angle = np.average([ft.direct_angle for ft in _fts], axis=0)
                _ft.pixel_pos = np.average([ft.pixel_pos for ft in _fts], axis=0)
                _ft.set_valid()
                _ft.u_gain = _gains.mean()
                _ft.gain = _gains.mean()
                _ft.id = self._alloc_frontier_id()
                _ft.parent_ids = list(set([pid for ft in _fts for pid in ft.parent_ids]))
                to_remove.extend([ft.id for ft in _fts])
                to_add.append((_ft, _ft.parent_ids))

        # Remove and add after loop
        self.remove_frontiers(to_remove)
        for _ft, _parent_ids in to_add:
            self.add_frontiers([_ft], parent_ids=_parent_ids)

        _n_valid_after = len(self.valid_frontiers)
        self.logger.debug(
            f"Merged frontiers from {_n_valid_before} to {_n_valid_after} valid frontiers."
        )

    def update_map(self, free_map=None, occ_map=None) -> None:
        """
        Update the planner's space with the occupancy (occ_map) and free space (free_map).
        At least one must be provided. No shape/dtype checks here by design.
        """
        if free_map is None and occ_map is None:
            raise ValueError("At least one of free_map or occ_map must be provided.")

        if free_map is not None:
            self.free_map = free_map
        if occ_map is not None:
            self.occ_map = occ_map

        # Only build KDTree if provided and non-empty
        free_for_kdt = free_map if (free_map is not None and len(free_map) > 0) else None
        occ_for_kdt  = occ_map  if (occ_map  is not None and len(occ_map)  > 0) else None
        self.planner.update_space(free_vx=free_for_kdt, occ_vx=occ_for_kdt)

        self.logger.debug(
            "Maps updated: free=%s, occ=%s",
            None if free_map is None else len(free_map),
            None if occ_map is None else len(occ_map),
        )

    def select_goal_frontier_id(self) -> int | None:
        """
        Select the frontier with the highest utility among valid frontiers.
        Sets and returns self.current_goal_ft_id. Returns None if none available.
        """
        fts = self.valid_frontiers
        if not fts:
            self.logger.debug("No valid frontiers available to select as goal.")
            self.current_goal_ft_id = None
            return None

        # Keep only finite utilities
        candidates = [(ft.id, float(ft.utility)) for ft in fts if np.isfinite(getattr(ft, "utility", np.nan))]
        if not candidates:
            self.logger.debug("No finite utilities available for goal selection.")
            self.current_goal_ft_id = None
            return None

        # Tie-break: utility desc, then u_gain desc (if present), then smaller ID
        def key_fn(fid_util):
            fid, util = fid_util
            ug = getattr(self.frontiers.get(fid), "u_gain", -np.inf)
            return (util, ug, -fid)

        fid, util = max(candidates, key=key_fn)
        self.current_goal_ft_id = fid
        self.logger.debug(f"Selected goal frontier ID: {fid} (utility={util:.6f}).")
        return fid
    
    def gain_adjustment(self, w_history_path: bool = True, w_map: bool = True) -> None:
        """
        Adjust each valid frontier's u_gain based on:
        1) proximity to past robot poses (history), and
        2) optional: visible free voxels from the frontier's camera pose (map).
        Keeps ft.gain as the base and writes ft.u_gain.
        """
        fts = self.valid_frontiers
        n = len(fts)
        if n == 0:
            self.logger.debug("No valid frontiers to adjust gains.")
            return

        # -------------------- ADJUST 1: history proximity --------------------
        if w_history_path:
            robot_ids = self.graph.get_node_R()
            if robot_ids:
                ft_W_T_C = np.stack([ft.pose6d for ft in fts], axis=0)  # (N,4,4)  W_T_C (C→W)
                robot_W_T_R = np.stack([self.robot_poses[rid] for rid in robot_ids], axis=0)  # (M,4,4)

                trans_diff, rot_diff = pose_difference(ft_W_T_C, robot_W_T_R)  # (N,M), (N,M)
                close_mask = (trans_diff < self.v_tras_thre) & (rot_diff < self.v_angl_thre)
                n_close = close_mask.sum(axis=1).astype(np.float32)  # (N,)
                reduction_1 = self.v_gain_reduction_factor * n_close
            else:
                reduction_1 = np.zeros(n, dtype=np.float32)
        else:
            reduction_1 = np.zeros(n, dtype=np.float32)

        # -------------------- ADJUST 2: map-based visibility --------------------
        voxel_vol = float(self.voxel_size ** 3)
        reduction_2 = np.zeros(n, dtype=np.float32)

        if w_map and (self.occ_map is not None) and (self.free_map is not None):
            # Frontier pose is W_T_C; for rendering we need C_T_W (extrinsics)
            C_T_W_batch = np.linalg.inv(np.stack([ft.pose6d for ft in fts], axis=0))  # (N,4,4)  C_T_W (W→C)

            depths = render_voxel_depth(
                occ_pts=self.occ_map,
                camera_extrinsics=C_T_W_batch,
                render_params={
                    "K": self.render_K,
                    "H": self.render_H,
                    "W": self.render_W,
                    "radius": self.voxel_size / 2,
                },
            )

            for i, (ft, C_T_W, depth) in enumerate(zip(fts, C_T_W_batch, depths)):
                # Count free voxels visible from this frontier view (respecting depth)
                vox_in_fov, _ = select_visible_points(
                    self.free_map,
                    K=self.render_K,
                    T=C_T_W,  # world→camera
                    img_size=(self.render_H, self.render_W),
                    max_depth=self.render_d_range,
                    min_depth_map=depth,
                )

                reduction_2[i] = float(vox_in_fov.shape[0]) * voxel_vol * float(self.render_decrease_factor)

                # Optional refinement: clamp base gain by max possible visible volume from this view
                if (ft.gain is not None) and not np.all(depth < 1e-6):
                    _, max_n_vis = compute_visible_voxels(
                        K=self.render_K,
                        T=C_T_W,  # world→camera
                        img_size=(self.render_H, self.render_W),
                        voxel_size=self.voxel_size,
                        max_depth=self.render_d_range,
                        min_depth_map=depth,
                    )
                    ft.gain = min(float(max_n_vis) * voxel_vol, float(ft.gain))


        self.logger.debug(
            "Before gain adjustment: " + ", ".join(f"{ft.id}: {ft.gain}" for ft in fts)
        )

        for ft, r1, r2 in zip(fts, reduction_1, reduction_2):
            base = float(ft.gain) if ft.gain is not None else 1e-4
            ft.u_gain = max(base - r1 - r2, 1e-4)

        self.logger.debug(
            "After gain adjustment: " + ", ".join(f"{ft.id}: {ft.u_gain}" for ft in fts)
        )

    def update_utility(self, current_pos) -> None:
        """
        Update each valid frontier's utility (Eq.(8) in the paper):
            utility = (u_gain ** utility_g_factor) / max(distance, 1e-6)
        """
        p = np.asarray(current_pos, dtype=float).reshape(3)

        valid = self.valid_frontiers
        for ft in valid:
            # Base gain: prefer u_gain, fall back to gain; clamp tiny positive
            base_gain = getattr(ft, "u_gain", None)
            if base_gain is None:
                base_gain = getattr(ft, "gain", 0.0)
            base_gain = float(max(base_gain, 1e-4))

            # Distance to current position (avoid divide-by-zero)
            d = float(np.linalg.norm(np.asarray(ft.pos3d, dtype=float) - p))
            denom = max(d, 1e-6)

            ft.utility = (base_gain ** float(self.utility_g_factor)) / denom

        self.logger.debug(f"Utility updated for {len(valid)} valid frontiers.")


    def filter_frontiers(self) -> None:
        """
        Mark frontiers invalid based on:
        1) bounding box (if set)
        2) min gain threshold
        3) view-direction z limit (if set)
        4) proximity to occupied space (if occ_map present)
        Then remove all invalid frontiers.
        """
        if not self.frontiers:
            self.logger.debug("No frontiers to filter.")
            return

        bbox = self.filter_bbox
        min_gain = float(self.filter_min_gain)
        max_vd_z = self.filter_max_vd_z
        check_occ = self.occ_map is not None

        for fid, ft in list(self.frontiers.items()):
            if not ft.is_valid:
                continue  # already invalid elsewhere

            # 1) Bounding box filter
            if bbox is not None:
                x, y, z = map(float, ft.pos3d)
                if not (bbox[0] <= x <= bbox[1] and
                        bbox[2] <= y <= bbox[3] and
                        bbox[4] <= z <= bbox[5]):
                    ft.set_invalid()
                    self.logger.debug(f"Frontier {fid} invalid (bbox).")
                    continue

            # 2) Gain filter
            g = ft.u_gain if ft.u_gain is not None else ft.gain
            g = float(g if g is not None else 0.0)
            if g < min_gain:
                ft.set_invalid()
                self.logger.debug(f"Frontier {fid} invalid (gain<{min_gain}).")
                continue

            # 3) View-direction z filter
            if max_vd_z is not None:
                vd_z = float(ft.view_direction[2])
                if abs(vd_z) > max_vd_z:
                    ft.set_invalid()
                    self.logger.debug(f"Frontier {fid} invalid (|vd_z|>{max_vd_z}).")
                    continue

            # 4) Too close to occupied space
            if check_occ and self.planner.isoccupied(ft.pos3d):
                ft.set_invalid()
                self.logger.debug(f"Frontier {fid} invalid (near occupied).")
                continue

        # Purge all marked
        self.remove_invalid_frontiers()


    def get_waypoints_from_graph(self, ft_id: int, robot_pose_id: int) -> np.ndarray:
        """
        Compute a final approach pose W_T_C for the target frontier, following the graph path
        from the given robot pose. Densifies with ~0.1 m spacing, finds the last reachable
        free-space point, and orients toward the next point (or the frontier if none).
        """
        # --- Validate inputs and fetch frontier ---
        assert ft_id in self.frontiers, f"Frontier ID {ft_id} not found."
        frontier = self.frontiers[ft_id]  # has .pos3d, .view_direction, .pose6d (W_T_C)

        # --- Shortest path (node ids R…→F) ---
        path, _ = self.graph.get_shortest_R_to_F(robot_id=robot_pose_id, frontier_id=ft_id)
        if not path:
            # No route; fall back to frontier’s own pose
            return frontier.pose6d

        # --- Coarse waypoints: robot node positions, then frontier position ---
        coarse_pts = []
        for node_id in path[:-1]:
            W_T_R = self.get_robot_pose(node_id)
            if W_T_R is None:
                raise KeyError(f"Robot pose for node {node_id} not found.")
            coarse_pts.append(W_T_R[:3, 3])
        coarse_pts.append(frontier.pos3d)

        # --- Densify path at ~0.1 m (no duplicate endpoints between segments) ---
        STEP = 0.1
        MAX_PER_SEG = 500
        fine_pts = []
        prev = None
        for cur in coarse_pts:
            cur = np.asarray(cur, dtype=float)
            if prev is None:
                prev = cur
                continue
            seg = cur - prev
            dist = float(np.linalg.norm(seg))
            if dist <= 1e-6:
                fine_pts.append(cur)
                prev = cur
                continue
            n = int(np.ceil(dist / STEP))
            n = min(max(n, 1), MAX_PER_SEG)
            # sample t in (0,1], i.e., exclude prev to avoid duplicates, include cur
            ts = np.linspace(1.0 / n, 1.0, n)
            pts = prev + ts[:, None] * seg
            fine_pts.extend(pts)
            prev = cur

        if not fine_pts:
            # No densification possible; use the frontier pose
            return frontier.pose6d

        # --- Reverse search: last reachable point (free and not occupied) ---
        break_idx = None
        for idx in range(len(fine_pts) - 1, -1, -1):
            pt = fine_pts[idx]
            if self.planner.isfree(pt) and not self.planner.isoccupied(pt):
                break_idx = idx
                break

        if break_idx is None:
            # Nothing along the path is free → go with the frontier pose
            return frontier.pose6d

        break_pt = np.asarray(fine_pts[break_idx], dtype=float)

        # Close enough to the frontier? keep its orientation, place at break_pt
        if np.linalg.norm(break_pt - frontier.pos3d) < self.v_tras_thre:
            W_T_C = frontier.pose6d.copy()
            W_T_C[:3, 3] = break_pt
            return W_T_C

        # --- Orientation at the break point ---
        # Prefer direction toward the next waypoint if there is one; otherwise toward the frontier.
        if break_idx < len(fine_pts) - 1:
            target = np.asarray(fine_pts[break_idx + 1], dtype=float)
        else:
            target = np.asarray(frontier.pos3d, dtype=float)

        direction = target - break_pt
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            # Degenerate → fall back to the frontier's view direction, or +Z
            vd = np.asarray(frontier.view_direction, dtype=float)
            vd_norm = float(np.linalg.norm(vd))
            direction = (vd / vd_norm) if vd_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            direction /= norm

        W_T_C = compute_alignment_transforms(
            origins=[break_pt],
            align_vec=direction,     # +Z aligns with travel direction
            align_axis=[0, 0, 1],
            appr_vec=[0, 0, -1],     # CV convention
            appr_axis=[0, 1, 0],
        )[0]
        W_T_C[:3, 3] = break_pt
        return W_T_C
    

    def plan_path_to_goal(
        self,
        current_pose: np.ndarray,
        interpolate_solution: bool = True,
        use_graph: bool = True,
    ) -> list[np.ndarray] | None:
        """
        Plan a path from the current robot pose W_T_R to the selected goal frontier.

        Args:
            current_pose: (4,4) W_T_R (R→W) robot pose in world frame.
            interpolate_solution: If True, densify the planner's solution.
            use_graph: If True, compute a goal pose via graph waypoints; else use the frontier pose.

        Returns:
            List of (4,4) poses along the path (world-frame), or None if planning fails.
        """

        # Select goal frontier
        goal_ft_id = self.select_goal_frontier_id()
        self.logger.debug(f"Planning path to goal frontier ID: {goal_ft_id}")
        if goal_ft_id is None:
            self.logger.debug("No goal frontier available for path planning.")
            return None
        goal_ft = self.frontiers[goal_ft_id]
        
        # Decide goal pose
        if use_graph:
            current_pose_id = self.add_robot_poses([current_pose])[0]  
            goal_pose = self.get_waypoints_from_graph(goal_ft_id, current_pose_id)

        else:
            goal_pose = self.get_frontier_pose(goal_ft)

        self.planner.update_start_goal(start=current_pose, goal=goal_pose)
        self.current_goal_pose = goal_pose  # Update the current goal pose to the last waypoint

        try:
            path_found = self.planner.solve(
                time_limit=self.max_planning_time,
                method=self.planning_algo
            )

            if not path_found:
                # If unreachable, drop this frontier as a goal candidate
                self.remove_frontiers([goal_ft_id])
                self.current_goal_ft_id = None
                self.current_goal_pose = None
                return None
        
            if interpolate_solution:
                self.planner.interpolate_path()

            return self.planner.get_solution_path(return_type="mat") 
            
        except Exception as e:
            self.logger.error(f"Path planning failed: {e}")
            # If planning fails, drop this frontier as a goal candidate
            self.remove_frontiers([goal_ft_id])
            self.current_goal_ft_id = None
            self.current_goal_pose = None
            self.logger.debug("Dropped goal frontier due to planning failure.")
            return None


    def get_graph_vis(self):
        """
        Get the visualization of the frontier graph.
        Returns:
            o3d geometry to visualize the graph.
            Frontiers as red spheres, robots as blue spheres, edges as thin cylinders.
        """
        ft_ids = self.graph.get_node_F()
        robot_ids = self.graph.get_node_R()
        FR_edges = self.graph.get_edge_FR() 
        RR_edges = self.graph.get_edge_RR()
        geometries = []
        # Visualize frontiers
        for ft_id in ft_ids:
            ft = self.frontiers[ft_id]
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.075)
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate(ft.pos3d)
            geometries.append(sphere)
        # Visualize robots/cameras
        for robot_id in robot_ids:
            pose = self.robot_poses[robot_id]
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.3, origin=[0, 0, 0]
            )
            coordinate_frame.transform(pose)
            geometries.append(coordinate_frame)
        # Visualize edges
        for edge in FR_edges:
            start_id, end_id, _ = edge
            start_pos = self.frontiers[start_id].pos3d if start_id in self.frontiers else self.robot_poses[start_id][:3, 3]
            end_pos = self.frontiers[end_id].pos3d if end_id in self.frontiers else self.robot_poses[end_id][:3, 3]
            cylinder = create_cylinder_between_points(start_pos, end_pos, radius=0.012,
                                                      color=[0, 0, 0])
            if cylinder is not None:
                geometries.append(cylinder)
        for edge in RR_edges:
            start_id, end_id, _ = edge
            start_pos = self.robot_poses[start_id][:3, 3]
            end_pos = self.robot_poses[end_id][:3, 3]
            cylinder = create_cylinder_between_points(start_pos, end_pos, radius=0.02,
                                                      color=[0.2, 0.8, 0.2])
            if cylinder is not None:
                geometries.append(cylinder)
        return geometries
    

    ## -- File I/O --
    @staticmethod
    def _json_default(o):
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    def write_to_file(self, file_path: str) -> None:
        """
        Append a JSON line snapshot of the manager state.
        """
        entry = {
            "all_frontiers": [ft.to_dict() for ft in self.all_frontiers],
            "valid_frontiers": [ft.to_dict() for ft in self.valid_frontiers],
            "robot_poses": {rid: pose.tolist() for rid, pose in self.robot_poses.items()},
            "current_robot_id": self._current_robot_id,
            "current_ft_goal_id": self.current_goal_ft_id if self.current_goal_ft_id is not None else None,
            "current_goal_pose": None if self.current_goal_pose is None else self.current_goal_pose.tolist(),
            "graph": self.graph.return_current_graph(),  
        }

        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(entry, default=self._json_default) + "\n")
            self.logger.debug(f"FrontierManager state appended to {file_path} (JSON lines).")
        except Exception as e:
            self.logger.error(f"Failed to write FrontierManager state to {file_path}: {e}")

    @staticmethod
    def read_from_file(file_path):
        """
        Read the state of the FrontierManager from a file.
        Args:
            file_path: Path to the file to read from.

        Returns:
            A List of entrys, each entry is a dictionary containing the state of the FrontierManager.
        """
        entries = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    entries.append(entry)
            return entries
        except Exception as e:
            print(f"Failed to read FrontierManager state from {file_path}: {e}")
            return []

