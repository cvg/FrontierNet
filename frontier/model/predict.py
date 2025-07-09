import torch
import logging
from frontier.model.utils.preprocess import preprocess
from frontier.model.unet import TwoHeadUnet


def load_model(path, num_classes=11, use_depth=True):

    model = TwoHeadUnet(
        classes=num_classes,
        in_channels=3 if not use_depth else 4,
        regression_downsample_factor=1.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model from: {path}")
    logging.info(f"Using device {device}")
    model.to(device=device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    logging.info("Model loaded! Num classes: {}".format(num_classes))

    return model


def predict_from_img(
    net,
    rgb_img,
    depth_img,
    device,
    scale_factor=0.5,
    use_depth=True,
    input_img_size=[224, 224],
):
    """
    Preprocess an RGB (and optionally depth) image pair, pack into a batch tensor,
    and run the model's predict method.
    """
    input_h, input_w = input_img_size
    if input_h != input_w:
        raise ValueError("input_img_size must be square, got {}".format(input_img_size))

    # Preprocess RGB
    rgb_tensor = (
        torch.from_numpy(
            preprocess(rgb_img, scale_factor, input_h, input_w, is_depth=False)
        )
        .unsqueeze(0)
        .to(device, dtype=torch.float32)
    )

    if use_depth:
        # Only preprocess depth when requested
        depth_tensor = (
            torch.from_numpy(
                preprocess(depth_img, scale_factor, input_h, input_w, is_depth=True)
            )
            .unsqueeze(0)
            .to(device, dtype=torch.float32)
        )
        inp = torch.cat((rgb_tensor, depth_tensor), dim=1)
    else:
        inp = rgb_tensor

    return net.predict(inp)
