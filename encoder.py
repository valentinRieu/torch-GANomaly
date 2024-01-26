import math

import numpy as np
import torch
from torch import Tensor, nn
from typing import Tuple

from torchviz import make_dot


class Encoder(nn.Module):
    """Encoder Network. Considering an input image, encodes a 'Latent vector' using Convolutional layers.
    We use a Feature Pyramid Network to encode the vector.

    Args:
        input_size (tuple[int, int]): Size of the input image (2 Dimensions)
        latent_size (int): Size of latent vector z (1 Dimension)
        n_input_channels (int): Number of input channels in the image (greyscale = 1, RGB = 3, CMYK = 4)
        n_features (int): Number of features per convolution layer
        hidden_size (int): Number of hidden convolutional layers excluding the Encoder the layer. Default =  0.
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 latent_size: int,
                 n_input_channels: int,
                 n_features: int,
                 hidden_size: int = 0
                 ):
        super().__init__()

        # We use Sequential models to stack our layers
        # This applies for the input layer, the hidden layers and Pyramid Network

        self.input_layers = nn.Sequential()

        self.input_layers.add_module(
            name=f'input-{n_input_channels}-{n_features}-conv',
            module=nn.Conv2d(n_input_channels, n_features, kernel_size=4, padding=4, bias=False)
        )
        # Standard activation layer using ReLU
        self.input_layers.add_module(
            name=f'input-{n_features}-relu',
            module=nn.ReLU(inplace=True)
        )

        # Hidden layers. They perform convolution and Batch Normalization
        # it should add stability the model
        # mean = 0, std = 1

        self.hidden_layers = nn.Sequential()

        for layer in range(hidden_size):
            self.hidden_layers.add_module(
                name=f'hidden-layer-{layer}-{n_features}-conv',
                module=nn.Conv2d(n_features, n_features, kernel_size=4, padding=1, bias=False)
            )
            self.hidden_layers.add_module(
                name=f'hidden-layer-{layer}-{n_features}-batchnorm',
                module=nn.BatchNorm2d(n_features)
            )
            self.hidden_layers.add_module(
                name=f'hidden-layer-{layer}-{n_features}-relu',
                module=nn.ReLU(inplace=True)
            )

        # Feature Pyramid Network. See README.md for graph of the architecture,
        # and Tsung-Yi Lin et al. 2017 (https://arxiv.org/abs/1612.03144)

        self.pyramid_features = nn.Sequential()
        curr_dim = min(input_size) // 2
        while curr_dim > 4:
            feature_in = n_features
            feature_out = n_features * 2
            self.pyramid_features.add_module(
                name=f'pyramid-{feature_in}-{feature_out}-conv',
                module=nn.Conv2d(feature_in, feature_out, kernel_size=4, padding=1, bias=False)
            )
            self.pyramid_features.add_module(
                name=f'pyramid-{feature_out}-batchnorm',
                module=nn.BatchNorm2d(feature_out)
            )
            self.pyramid_features.add_module(
                name=f'pyramid-{feature_out}-relu',
                module=nn.ReLU(inplace=True)
            )
            n_features = feature_out
            curr_dim //= 2



    def forward(self, input_tensor: Tensor) -> Tensor:
        """ Computation performed at every call"""
        output = self.input_layers(input_tensor)
        output = self.hidden_layers(output)
        output = self.pyramid_features(output)
        return output


if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 256, 256)

    encoder = Encoder(
        (256, 256),
        8,
        3,
        4,
        4
    )

    output = encoder(dummy_input)
    dot = make_dot(output, params=dict(encoder.named_parameters()))
    dot.graph_attr['dpi'] = '1000'
    dot.render('model_graph', format='png')

