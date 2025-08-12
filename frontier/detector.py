import numpy as np
import torch
import hdbscan
import logging
from typing import Optional, List, Any, Tuple

from frontier.model.utils.postprocess import prediction2frontiermap
from frontier.model.utils.preprocess import preprocess
from frontier.model.predict import predict_from_img
from frontier.frontier import Frontier
from frontier.base import Base
from utils.geometry import (
    compute_gradient,
    grad_mag_and_direct_from_gradmap,
    avg_depth_from_bin_mask,
    direction_on_image_to_vec_in_world,
    find_center_point,
)


class FrontierDetector(Base):
    def __init__(
        self,
        model: Any,
        camera_intrinsic: np.ndarray,
        use_depth: bool = True,
        img_size_model: Tuple[int, int] = (320, 320),
        device: str = "cuda",
        log_level: int = logging.INFO,
    ):
        """
        Args:
            model: the neural network or algorithm instance
            camera_intrinsic: original RGB camera intrinsic matrix
            use_depth: whether to use the depth image
            img_size_model: (width, height) expected by the model
            device: "cuda" or "cpu"
            log_level: logging level (e.g. logging.INFO)
        """
        # configure logging
        super().__init__(params=None, log_level=log_level)

        # core components
        self.model = model
        self.device = torch.device(device if device == "cuda" else "cpu")

        # intrinsics & preprocessing state
        self.ori_intrin: np.ndarray = camera_intrinsic
        self.scale_factor: Optional[float] = None
        self.pro_intrin: Optional[np.ndarray] = None
        self.img_size_model: Tuple[int, int] = img_size_model

        # modality flags
        self.use_depth: bool = use_depth

        # inputs
        self.raw_rgb: Optional[np.ndarray] = None
        self.raw_depth: Optional[np.ndarray] = None
        self.extrinsic: Optional[np.ndarray] = None

        # outputs
        self.df: Optional[np.ndarray] = None  # distance field
        self.ft_region: Optional[np.ndarray] = None  # frontier region mask
        self.info_gain: Optional[np.ndarray] = None
        self.ft_3D: Optional[np.ndarray] = None  # 3D frontier clusters

    def _cal_processed_intrinsic(self, input_img_size):
        """
        calculate new intrinsic matrix after preprocessing
        which includes scalling and center cropping
        specifically:
            f' = f * scale_factor, c' = c * scale_factor
            cx'' = cx' - delta_x and cy'' = cy' - delta_y
        """
        W, H = input_img_size
        model_W, model_H = self.img_size_model

        # scaled image dims
        scaled_W, scaled_H = self.scale_factor * W, self.scale_factor * H

        # cropping offsets
        offset_x = (scaled_W - model_W) / 2
        offset_y = (scaled_H - model_H) / 2

        # compute new intrinsic matrix
        K = self.ori_intrin * self.scale_factor
        K[0, 2] -= offset_x
        K[1, 2] -= offset_y

        # cache and return
        self.pro_intrin = K

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        df_normalizer: float = 3.0,
        df_thr: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run frontier detection on an RGB + depth pair.

        Returns:
            ft_region: 2D frontier-region mask
            info_gain:  2D info-gain map
        """
        # store raw inputs
        self.raw_rgb, self.raw_depth = rgb, depth
        H, W = rgb.shape[:2]
        self.logger.debug(f"Input size (WxH): {W}x{H}")
        self.scale_factor = (
            max(self.img_size_model[0] / H, self.img_size_model[1] / W) + 0.02
        )

        # compute scale & intrinsics
        self._cal_processed_intrinsic((W, H))

        # --- 1) Inference ---
        self.logger.debug("Running model inference...")
        df_tensor, cls_mask = predict_from_img(
            net=self.model,
            rgb_img=rgb,
            depth_img=depth,
            device=self.device,
            scale_factor=self.scale_factor,
            use_depth=self.use_depth,
            input_img_size=self.img_size_model,
        )
        self.df = df_tensor.cpu().detach().numpy().squeeze()
        self.logger.debug("Inference complete; DF shape: %s", self.df.shape)

        # --- 2) Postprocessing ---
        self.logger.debug(
            "Postprocessing: normalizer=%.2f, threshold=%.2f", df_normalizer, df_thr
        )
        self.ft_region, self.info_gain = prediction2frontiermap(
            df=df_tensor,
            cls_mask=cls_mask,
            n_classes=self.model.n_classes,
            df_normalizer=df_normalizer,
            threshold=df_thr,
        )

        return self.ft_region, self.info_gain

    def anchor_fts(
        self, depth: np.ndarray, extrinsic: np.ndarray
    ) -> Optional[List[Frontier]]:
        """
        Anchoring 2D Frontiers to 3D Frontier Clusters using depth and camera extrinsics.
        """
        # 1) Preprocess depth (resize + center crop)
        depth = preprocess(
            depth,
            self.scale_factor,
            *self.img_size_model,
            is_depth=True,
            normalize_depth=False,
        ).squeeze()
        self.extrinsic = extrinsic

        # 2) Validate shapes: ft_region and depth should align
        if self.ft_region.shape != depth.shape:
            raise ValueError(f"Shape mismatch: {self.ft_region.shape} vs {depth.shape}")

        # 3) Get depth feature: direction & avg depth
        direction, depth_avg = self.get_depth_feature(self.ft_region, depth)

        # 4) Get per-pixel 2D frontier features, namely Ft^{2D} in the paper
        self.get_ft_feature(direction, depth_avg)

        # 5) get frontier clusters
        result = self.get_3D_ft_clusters(depth_avg, extrinsic)
        if result is None:
            return None
        ft_clusters, _, _ = result  # (Ft^{3D} in the paper)

        # 6) Build Frontier objects from each 3D cluster
        frontiers = []
        for cluster in ft_clusters:
            # [u, v, dx, dy, gain, …, x, y, z, vx, vy, vz]
            u, v, dx, dy, gain, *_, x, y, z, vx, vy, vz = cluster[:12]
            f = Frontier()
            f.pixel_pos = (u, v)
            f.direct_angle = np.arctan2(dy, dx)
            f.gain = f.u_gain = gain
            f.pos3d = (x, y, z)
            f.view_direction = (vx, vy, vz)
            f.set_valid()
            frontiers.append(f)

        self.ft_3D = frontiers
        return frontiers

    def get_depth_feature(
        self, bin_mask: np.ndarray, depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        get 2D viewing direction and average depth for each frontier pixel
        this corresponding to Sec. III.E.1 and Sec. III.E.3 in the paper
        """
        # quick sanity check
        if bin_mask.shape != depth.shape:
            raise ValueError(
                f"bin_mask shape {bin_mask.shape} != depth shape {depth.shape}"
            )

        # process depth image, get gradian and direction
        grad_x, grad_y = compute_gradient(depth, kernel_size=9)
        magnitude, direction = grad_mag_and_direct_from_gradmap(grad_x, grad_y)
        self.logger.debug(
            "Gradient computed: grad_x %s, grad_y %s, direction %s",
            grad_x.shape,
            grad_y.shape,
            direction.shape,
        )

        # get avg depth
        avg_depth = avg_depth_from_bin_mask(
            bin_mask=bin_mask,
            depth=depth,
            gradient_direction=direction,
            gradient_magnitude=magnitude,
            skip_close_points=0.3,
        )
        self.logger.debug("Average depth computed, shape %s", avg_depth.shape)

        return direction, avg_depth

    def get_ft_feature(self, direction: np.ndarray, depth_avg: np.ndarray) -> None:
        """construct per-pixel features from detection"""
        # find all frontier pixel coordinates
        ys, xs = np.where(self.ft_region == 1)

        # filter out invalid frontier pixels (frontier pixels with outlier depth)
        valid_mask = depth_avg[ys, xs] != 0
        ys, xs = ys[valid_mask], xs[valid_mask]

        # number of valid frontier pixels
        n = len(ys)
        # preallocate feature array: [x, y, cos(theta), sin(theta), gains, depth_avg]
        feature2D = np.zeros((n, 6), dtype=float)

        # normalized pixel coordinates
        feature2D[:, 0] = xs / self.img_size_model[1]  # x
        feature2D[:, 1] = ys / self.img_size_model[0]  # y

        # angles in radians
        radians = direction[ys, xs] * np.pi / 180
        feature2D[:, 2] = np.cos(radians)  # cos(theta)
        feature2D[:, 3] = np.sin(radians)  # sin(theta)

        # gains and depth
        feature2D[:, 4] = self.info_gain[ys, xs]
        feature2D[:, 5] = depth_avg[ys, xs]

        # assign back to instance
        self.feature2D = feature2D

    def get_3D_ft_clusters(
        self,
        depth_avg: np.ndarray,
        cam_extrinsic: np.ndarray,
        remove_close_ft_thr: float = 0.3,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """
        cluster and lift 2D frontier features to 3D frontiers
        this corresponds to Sec. III.E.2 in the paper
        """
        # set up clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=20,
            min_samples=25,
            cluster_selection_epsilon=0.5,
            metric="l2",
        )
        cluster_weights = np.array([10, 10, 1, 1, 0.3])
        cluster_dims = [0, 1, 2, 3, 5]

        try:
            # weight and select dims in one go
            feats = self.feature2D[:, cluster_dims] * cluster_weights
            labels = clusterer.fit_predict(feats)
            max_label = labels.max()
            num_clusters = int(max_label + 1)
        except Exception:
            self.logger.debug("No frontiers detected")
            return None

        # precompute camera-to-world
        T_cam2world = np.linalg.inv(cam_extrinsic)
        clusters_out = []

        # process each cluster
        for lbl in range(num_clusters):
            mask = labels == lbl
            if not mask.any():
                continue
            pts = self.feature2D[mask]

            _feature = np.zeros(12, dtype=float)

            # (x, y) center point (closest to centroid)
            _feature[0], _feature[1] = find_center_point(pts[:, :2])

            # mean direction unit vector
            vec_sum = pts[:, 2:4].sum(axis=0)
            norm = np.linalg.norm(vec_sum)
            if norm == 0:
                continue
            _feature[2:4] = vec_sum / norm

            # robust gain
            gains = np.sort(pts[:, 4])
            mid = len(gains) // 2
            if len(gains) < 10:
                _feature[4] = gains.mean()
            else:
                _feature[4] = np.median(gains[mid : int(0.9 * len(gains))])
            _feature[4] *= (
                10 * 0.001
            )  # recover volume NOTE: this scaling is to compensate for the volume scaling in the detection model

            # robust depth (ignore zeros)
            depths = np.sort(pts[:, 5])
            depths = depths[depths > 0]
            if depths.size == 0:
                continue
            low, high = int(0.05 * len(depths)), int(0.9 * len(depths))
            _feature[5] = np.median(depths[low:high])

            # Skip if too close
            if _feature[5] < remove_close_ft_thr:
                continue

            # backproject center pixel to 3D
            H, W = self.img_size_model
            i_y = int(_feature[0] * W)
            i_x = int(_feature[1] * H)
            z = depth_avg[i_x, i_y]
            x = (i_y - self.pro_intrin[0, 2]) * z / self.pro_intrin[0, 0]
            y = (i_x - self.pro_intrin[1, 2]) * z / self.pro_intrin[1, 1]
            cam_pt = np.array([x, y, z, 1.0]).reshape(4, 1)
            world_pt = T_cam2world @ cam_pt
            _feature[6:9] = world_pt[:3, 0]

            # Project 2D direction to 3D
            ang = np.arctan2(_feature[3], _feature[2])
            _feature[9:12] = direction_on_image_to_vec_in_world(
                ang, self.pro_intrin, cam_extrinsic
            ).ravel()

            clusters_out.append(_feature)

        if not clusters_out:
            return None

        ft_clusters = np.vstack(clusters_out)
        return ft_clusters, labels, ft_clusters.shape[0]

    def save_result_npz(self, save_path):
        """
        save the result to a npz file
        """
        np.savez(
            save_path,
            raw_rgb=self.raw_rgb,
            raw_depth=self.raw_depth,
            input_img_size=self.img_size_model,
            df=self.df,
            ft_region=self.ft_region,
            info_gain=self.info_gain,
            ft_3D=self.ft_3D,
            pro_intrinsic=self.pro_intrin,
            ori_intrinsic=self.ori_intrin,
            extrinsic=self.extrinsic,
        )

    def get_detection_result(self):
        """
        return the result as a dictionary
        """
        return {
            "raw_rgb": self.raw_rgb,
            "raw_depth": self.raw_depth,
            "input_img_size": self.img_size_model,
            "df": self.df,
            "ft_region": self.ft_region,
            "info_gain": self.info_gain,
            "ft_3D": self.ft_3D,
            "pro_intrinsic": self.pro_intrin,
            "ori_intrinsic": self.ori_intrin,
            "extrinsic": self.extrinsic,
        }
