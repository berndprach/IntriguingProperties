
import torch
from torch import nn


class BatchCentering2d(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum

        # Buffer: not updated by optimizer but saved in state_dict.
        self.register_buffer("running_mean", torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=(0, 2, 3))

        if self.training:
            with torch.no_grad():
                mom = self.momentum
                self.running_mean = self.running_mean * (1-mom) + mean * mom
            return x - mean[None, :, None, None]
        else:
            return x - self.running_mean[None, :, None, None]


class BatchCentering1d(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum

        self.register_buffer("running_mean", torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=0)

        if self.training:
            with torch.no_grad():
                mom = self.momentum
                self.running_mean = self.running_mean * (1-mom) + mean * mom
            return x - mean[None, :]
        else:
            return x - self.running_mean[None, :]


class Affine2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        linear = x * self.weight[None, :, None, None]
        return linear + self.bias[None, :, None, None]


class Affine1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        linear = x * self.weight[None, :]
        return linear + self.bias[None, :]


class BCAffine2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.batch_centering = BatchCentering2d(num_features)
        self.affine = Affine2d(num_features)

    def forward(self, x):
        return self.affine(self.batch_centering(x))


class LayerNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / std * self.weight[None, :] + self.bias[None, :]


class PlainLayerNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        if self.training:
            return (x - mean) / std
        else:
            return x - mean
