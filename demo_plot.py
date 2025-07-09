import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path
from utils.vis_utils import (
    visualize_distance_field,
    visualize_gain_map,
    visualize_2D_frontier,
    visualize_3D_frontier,
)
from frontier.model.utils.preprocess import resize_centercrop_img


def plot_result(result_path):
    """
    Load FrontierDetector results from a .npz file, generate 2D overlays
    (distance field, info-gain, frontier mask) side-by-side with the
    preprocessed RGB, display them, and then invoke the 3D frontier visualization.
    """
    # load and validate
    if not Path(result_path).is_file() or not result_path.endswith(".npz"):
        raise FileNotFoundError(f"Result file not found: {result_path}")
    data = np.load(result_path, allow_pickle=True)

    # unpack
    rgb: np.ndarray = data["raw_rgb"]  # (H, W, 3), uint8
    depth: np.ndarray = data["raw_depth"]  # (H, W), float32
    input_size = tuple(data["input_img_size"])  # (model_H, model_W)
    df: np.ndarray = data["df"]
    info_gain: np.ndarray = data["info_gain"]
    ft_3D: np.ndarray = data["ft_3D"]
    ori_intrinsic: np.ndarray = data["ori_intrinsic"]
    extrinsic: np.ndarray = data["extrinsic"]

    # preprocess RGB to model input size + BGR for OpenCV
    H, W = rgb.shape[:2]
    model_H, model_W = input_size
    scale = max(model_H / H, model_W / W) + 0.02
    pil_img = Image.fromarray(rgb)
    preprocessed = resize_centercrop_img(pil_img, scale, input_size)
    pre_bgr = np.array(preprocessed)[..., ::-1].copy()  # RGB→BGR

    # build 2D visualizations
    vis_df = visualize_distance_field(df, pre_bgr, cmap_name="turbo")
    vis_info = visualize_gain_map(
        info_gain,
        df,
        pre_bgr,
        cmap_name="afmhot",
        smooth=0.3,
        dilate_kernel=3,
        blur_kernel=1,
    )
    vis_ft2d = visualize_2D_frontier(ft_3D, pre_bgr, input_size)

    # concatenate and display
    combined = np.concatenate([pre_bgr, vis_df, vis_info, vis_ft2d], axis=1)
    window = "RGB | DF | InfoGain | 2D Frontier"
    print("Displayed 2D overlays. Press any key to continue to 3D visualization...")
    cv2.imshow(window, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 3D visualization
    visualize_3D_frontier(
        ft_3D,
        intrinsic=ori_intrinsic,
        extrinsic=extrinsic,
        rgb=rgb,
        depth=depth,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize FrontierNet results")
    parser.add_argument(
        "--result_path", type=str, required=True, help="Path to the result .npz file"
    )
    args = parser.parse_args()
    result_path = args.result_path
    plot_result(result_path)
