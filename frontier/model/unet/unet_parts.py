import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base.modules import Activation


class ScaledTanh(nn.Module):
    def __init__(self):
        super(ScaledTanh, self).__init__()

    def forward(self, x):
        return 0.5 * (torch.tanh(x) + 1.0)


class ClampReLU(nn.Module):
    def __init__(self):
        super(ClampReLU, self).__init__()

    def forward(self, x):
        return F.relu(x).clamp(min=0, max=1)


class PredictionModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        init.initialize_head(self.df_regression_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        self.check_input_shape(x)

        # do the padding if needed
        h, w = x.shape[-2:]
        if h % self.output_stride != 0 or w % self.output_stride != 0:
            new_h = (
                (h // self.output_stride + 1) * self.output_stride
                if h % self.output_stride != 0
                else h
            )
            new_w = (
                (w // self.output_stride + 1) * self.output_stride
                if w % self.output_stride != 0
                else w
            )
            x = nn.functional.pad(x, (0, new_w - w, 0, new_h - h))

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        df = self.df_regression_head(decoder_output)

        if h % self.output_stride != 0 or w % self.output_stride != 0:
            masks = masks[:, :, :h, :w]
            df = df[
                :,
                :,
                : int(h * self.df_regression_head.downsample_factor),
                : int(w * self.df_regression_head.downsample_factor),
            ]

        return df, masks

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        x = self.forward(x)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        super().__init__(
            # Block 1
            nn.Conv2d(
                in_channels, 64, kernel_size=kernel_size, padding=kernel_size // 2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Block 2
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Output
            nn.Conv2d(64, out_channels, kernel_size=1),
            # Optional Upsampling
            nn.Upsample(scale_factor=upsampling) if upsampling > 1 else nn.Identity(),
            # Activation from SMP
            Activation(activation),
        )


class DfRegressionHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, downsample_factor, kernel_size=3):
        super().__init__(
            # Block 1
            nn.Conv2d(
                in_channels, 64, kernel_size=kernel_size, padding=kernel_size // 2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Block 2
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Output
            nn.Conv2d(64, out_channels, kernel_size=1),
            # Downsampling layer
            (
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=int(1 / downsample_factor),
                    padding=1,
                )
                if downsample_factor != 1
                else nn.Identity()
            ),
            nn.ReLU(),
        )


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
