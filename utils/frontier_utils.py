import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml


def ft_pos_direct_distance(ft1, ft2, weights=[1, 1]):
    """Function to compute the custom distance for 3d position and 3d direction"""
    """this is for 3d frontier clustering in ft manager"""

    # Split the feature vectors: first 3 are 3D position, next 3 are 3D direction
    pos1, vd1 = ft1[:3], ft1[3:]
    pos2, vd2 = ft2[:3], ft2[3:]
    # L2 (Euclidean) distance for 3D position
    # pos_distance = euclidean(pos1, pos2)

    # L1 (Manhattan) distance for 3D position
    pos_distance = np.sum(np.abs(pos1 - pos2))

    # Cosine similarity for 3D direction
    vd1_norm = vd1 / (np.linalg.norm(vd1) + 1e-6)
    vd2_norm = vd2 / (np.linalg.norm(vd2) + 1e-6)
    direction_similarity = np.dot(vd1_norm, vd2_norm)
    # Convert cosine similarity to a distance-like measure
    direction_distance = (
        1 - direction_similarity
    )  # 1 - cos(theta) gives a distance-like value

    # Combine the two distances (this can be tuned)
    combined_distance = weights[0] * pos_distance + weights[1] * direction_distance

    return combined_distance


def read_config_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def draw_marks(
    img: np.ndarray,
    points: list[tuple[int, int]],
    labels: list[str] = None,
    radius: int = 60,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Draw Bumble-style visual markers: semi-transparent gray fill,
    black border, and black label inside the circle.
    """
    # Convert to PIL for nicer font rendering
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil, "RGBA")

    if labels is None:
        labels = [chr(65 + i) for i in range(len(points))]  # A, B, C...

    for (x, y), label in zip(points, labels):
        # Draw gray filled circle with alpha
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=(128, 128, 128, int(255 * alpha)),
            outline=(0, 0, 0, 255),
            width=2,
        )

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(radius * 1.2))
        except:
            font = ImageFont.load_default()

        if hasattr(font, "getbbox"):
            bbox = font.getbbox(label)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            offset_x = bbox[0]
            offset_y = bbox[1]
        else:
            text_width, text_height = font.getsize(label)
            offset_x = offset_y = 0

        text_pos = (x - text_width // 2 - offset_x, y - text_height // 2 - offset_y)

        draw.text(text_pos, label, fill=(0, 0, 0, 255), font=font)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
