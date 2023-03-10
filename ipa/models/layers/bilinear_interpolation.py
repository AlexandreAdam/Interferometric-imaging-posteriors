import torch
from torch import nn


class BilinearInterpolation(nn.Module):
    def __init__(self):
        super(BilinearInterpolation, self).__init__()

    def __call__(self, images, coordinates):
        return self.forward(images, coordinates)

    def forward(self, images, coordinates):
        return self.interpolate(images, coordinates)

    @staticmethod
    def interpolate(images, coordinates):
        B, C, H, W = images.shape
        x, y = torch.tensor_split(coordinates, 2, dim=1)
        x = x.squeeze().view(-1)
        y = y.squeeze().view(-1)
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clip(x0, 0, W - 1)
        x1 = torch.clip(x1, 0, W - 1)
        y0 = torch.clip(y0, 0, H - 1)
        y1 = torch.clip(y1, 0, H - 1)
        x = torch.clip(x, 0, W - 1)
        y = torch.clip(y, 0, H - 1)

        Ia = images[..., x0, y0]
        Ib = images[..., x0, y1]
        Ic = images[..., x1, y0]
        Id = images[..., x1, y1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        _, _, new_H, new_W = coordinates.shape
        return (wa * Ia + wb * Ib + wc * Ic + wd * Id).view(B, C, new_H, new_W)
