import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from frontier.detector import FrontierDetector
from frontier.model.predict import load_model
from mono_depth.Metric3D import metric_depth_from_rgb as metric_depth_from_rgb_metric3d
from mono_depth.UniK3D import metric_depth_from_rgb as metric_depth_from_rgb_unik3d
from utils.frontier_utils import read_config_yaml

# suppress verbose matplotlib logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an RGB image and return a (H,W,3) float32 array in [0,1].
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Input must be a .jpg/.jpeg/.png file")

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def estimate_depth(
    rgb_uint8: np.ndarray, intrinsics: np.ndarray, model_name: str, H: int, W: int
) -> np.ndarray:
    """
    Estimate a metric depth map from an RGB image.
    """
    if model_name == "Metric3D":
        logging.info("Using Metric3D depth model")
        return metric_depth_from_rgb_metric3d(
            rgb_input=rgb_uint8,
            intrinsic_mat=intrinsics,
            camera_W=W,
            camera_H=H,
            local_model_path=None,
        )
    elif model_name == "UniK3D":
        logging.info("Using UniK3D depth model")
        return metric_depth_from_rgb_unik3d(
            rgb_input=rgb_uint8, intrinsic_mat=intrinsics
        )
    else:
        raise ValueError(f"Unsupported depth model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run frontier detection from a single RGB image"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--input_img", type=Path, required=True, help="Path to input image (.jpg/.png)"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("examples/"),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--depth_source",
        choices=["Metric3D", "UniK3D"],
        default="UniK3D",
        help="Depth estimation model",
    )
    parser.add_argument(
        "--unet_weight",
        type=Path,
        default=Path("model_weights/rgbd_11cls.pth"),
        help="Path to UNet model weights",
    )
    args = parser.parse_args()

    # set up root logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    # load configuration
    config = read_config_yaml(args.config)

    # load and normalize image
    image = load_image(args.input_img)
    H, W = image.shape[:2]
    image_name = args.input_img.stem
    logger.info("Loaded image %s (%dx%d)", image_name, W, H)

    # compute principal point
    cx = config.get("cx", W / 2 - 0.5)
    cy = config.get("cy", H / 2 - 0.5)
    fx = fy = config["focal_length"]
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    logger.info("Using intrinsics:\n%s", intrinsics)

    # load depth model
    unet = load_model(
        path=args.unet_weight, num_classes=config["num_classes"], use_depth=True
    )

    # estimate depth
    rgb_uint8 = (image * 255).astype(np.uint8)
    depth = estimate_depth(rgb_uint8, intrinsics, args.depth_source, H, W)
    logger.info("Depth estimated using %s model", args.depth_source)

    # initialize detector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = FrontierDetector(
        model=unet,
        camera_intrinsic=intrinsics,
        use_depth=True,
        img_size_model=tuple(config["input_img_size"]),
        device=device,
    )

    # run detection
    _, _ = detector.detect(
        rgb=rgb_uint8,
        depth=depth,
        df_normalizer=config["df_normalizer"],
        df_thr=config["df_thr"],
    )
    logger.info("2D frontier detection complete")

    # Get Frontiers in 3D
    # a default extrinsic matrix for a front-facing camera
    extrinsic = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    frontiers = detector.anchor_fts(depth=depth, extrinsic=extrinsic)
    logger.info("Get %d frontiers to 3D", len(frontiers) if frontiers else 0)

    # save results to npz
    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / f"{image_name}_ft.npz"
    detector.save_result_npz(save_path=str(result_path))
    logger.info("Saved frontier data to %s", result_path)


if __name__ == "__main__":
    main()
