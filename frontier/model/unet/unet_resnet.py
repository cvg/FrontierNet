from typing import Optional, Union, List
from segmentation_models_pytorch.encoders import get_encoder
from .unet_parts import UnetDecoder, PredictionModel, SegmentationHead, DfRegressionHead
import torch.nn as nn
import torch


class TwoHeadUnet(PredictionModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        regression_downsample_factor: float = 1.0,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # Modify the first convolutional layer to accept 4 channels
        if in_channels != 3:
            self.encoder.conv1 = self.modify_first_conv(self.encoder.conv1, in_channels)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.df_regression_head = DfRegressionHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            downsample_factor=regression_downsample_factor,
            kernel_size=3,
        )

        self.name = "u-{}".format(encoder_name)
        self.initialize()

        self.output_stride = 32
        self.n_classes = classes
        self.n_channels = in_channels

    @staticmethod
    def modify_first_conv(conv_layer, in_channels):
        if in_channels == conv_layer.in_channels:
            return conv_layer
        else:
            new_conv = nn.Conv2d(
                in_channels,
                conv_layer.out_channels,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.padding,
                bias=(conv_layer.bias is not None),
            )

            # Adjust weights for new in_channels
            with torch.no_grad():
                if in_channels < conv_layer.in_channels:
                    new_conv.weight.data = conv_layer.weight.data[:, :in_channels, :, :]
                else:
                    new_conv.weight.data[:, : conv_layer.in_channels, :, :] = (
                        conv_layer.weight.data
                    )
                    new_conv.weight.data[:, conv_layer.in_channels :, :, :] = (
                        conv_layer.weight.data[
                            :, : in_channels - conv_layer.in_channels, :, :
                        ]
                    )

            return new_conv
