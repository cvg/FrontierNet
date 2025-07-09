import numpy as np
from PIL import Image
from torchvision import transforms


def resize_centercrop_img(img, scale, img_size):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    # Resize the image
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BICUBIC)

    # Center crop the image
    center_crop = transforms.CenterCrop((img_size[0], img_size[1]))
    img = center_crop(img)

    return img


def preprocess(
    img_in, scale, input_height, input_width, is_depth=False, normalize_depth=True
):
    # check if input image is a numpy array
    assert isinstance(img_in, np.ndarray), "Input image should be a numpy array"

    img = resize_centercrop_img(img_in, scale, (input_width, input_height))
    img = np.asarray(img).astype(np.float32)

    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))

    if is_depth:
        # depth should be with only one channel
        if not img.shape[0] == 1:
            # make it with only one channel but keep the same dimension (1, H, W)
            img = img[0:1, ...]
        # normalize depth
        if normalize_depth:
            # Normalize depth to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    else:
        if (img > 1).any():
            img = img / 255.0

    return img
