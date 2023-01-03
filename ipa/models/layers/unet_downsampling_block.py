import torch
import torch.nn as nn
from ipa.definitions import DTYPE
from ipa.models.utils import get_activation
from .spectral_normalization import SpectralNorm
from .conv2dsame import Conv2dSame


class UnetDownsamplingBlock(nn.Module):
    """
    Classical downsampling Unet block, with strided convolution for downsampling
    """
    def __init__(
            self,
            in_channels: int,
            channels: int,
            layers: int = 2,
            stride: int = 2,
            non_linearity: str = "elu",
            kernel_size: int = 3,
            spectral_norm: bool = False,
            batch_norm: bool = False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super(UnetDownsamplingBlock, self).__init__()
        self.stride = stride
        self.non_linearity = get_activation(non_linearity)
        if spectral_norm:
            sp_norm = SpectralNorm
        else:
            sp_norm = lambda x: x
        if batch_norm:
            bn_norm = nn.BatchNorm2d
        else:
            bn_norm = lambda **kwargs: nn.Identity()
        bn_params = {"device": device, "dtype": DTYPE}
        conv_params = {"bias": False, "device": device, "dtype": DTYPE, "padding": "same"}
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.extend(
                [
                    sp_norm(nn.Conv2d(in_channels=in_channels if i==0 else channels, out_channels=channels, kernel_size=kernel_size, stride=1, **conv_params)),
                    bn_norm(num_features=channels, **bn_params)
                ]
            )
        self.downsampling_layer = sp_norm(Conv2dSame(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, **conv_params))
        self.shortcut = nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.non_linearity(x)
        skip = self.shortcut(x)
        x = self.downsampling_layer(x)
        x = self.non_linearity(x)
        return x, skip


if __name__ == '__main__':
    x = torch.randn([10, 4, 8, 8])
    l = UnetDownsamplingBlock(4, 4)
    l(x)