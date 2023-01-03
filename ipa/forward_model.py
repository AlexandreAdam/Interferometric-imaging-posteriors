import torch
import torchkbnufft as tkbn
from ipa.definitions import DEVICE


class ForwardModel:
    # TODO verify how we will apply the phase to this, or if there is a phase correction needed
    def __init__(self, uv_points: torch.Tensor, pixels: int, kernel="kaiser_bessel", device=DEVICE, **kwargs):
        self.pixels = pixels
        self.uv_points = uv_points.to(torch.cfloat)  # TODO make sure this is in radians / voxels
        if self.uv_points.shape[0] != 2:
            assert self.uv_points.shape[1] == 2
            self.uv_points = self.uv_points.T
        if kernel == "keiser_bessel":
            self.nufft_object = tkbn.KbNufft(im_size=[pixels, pixels], device=DEVICE, **kwargs)
            self.nufft = lambda x: self.nufft_object.forward(x, self.uv_points)
        elif kernel == "toeplitz":
            self.nufft_object = tkbn.KbNufft(im_size=[pixels, pixels], device=DEVICE, **kwargs)
            self.kernel = tkbn.calc_toeplitz_kernel(self.uv_points, [pixels, pixels])
            self.nufft = lambda x: self.nufft_object(x, self.kernel)

    def forward(self, x) -> torch.Tensor:
        return self.nufft(x)
