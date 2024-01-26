import math
import torch
from torch import Tensor, nn
from typing import Tuple


class Decoder(nn.Module):
    """Decoder Network. Considering an input vector, tries to decode and generate the original image.
    Uses transposed convolutional layers to reach the image dimension.

    Args:
        input_size (tuple[int, int]): Size of the output image
        latent_size (int): Size of the latent vector z (input of the neural network)
        n_input_channels (int): Number of input channels in the image (greyscale = 1, RGB = 3, CMYK = 4)
        n_features (int): Number of features per convolution layer
        hidden_size (int): Number of hidden  trconvolutional layers. Default is 0.
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 latent_size: int,
                 n_input_channels: int,
                 n_features: int,
                 hidden_size: int = 0
                 ):
        super().__init__()

        # Inverse feature pyramid layers using transposed convolution layers. See README.md
        #
        # We will consider the smallest dimension of the image
        exp_factor = math.ceil(math.log(min(input_size) // 2, 2))

        n_input_features = n_features * (2**exp_factor)

        self.pyramid_features = nn.Sequential()

        curr_dim = min(input_size) // 2

        while curr_dim > 4:
            feature_in = n_input_features
            feature_out = n_input_features // 2
            self.inverse_pyramid.add_module(
                name=f"pyramid-{feature_in}-{feature_out}-convt",
                module=nn.ConvTranspose2d(feature_in, feature_out, kernel_size=4, padding=1, bias=False)
            )
            self.inverse_pyramid.add_module(f"pyramid-{feature_out}-batchnorm", nn.BatchNorm2d(feature_out))
            self.inverse_pyramid.add_module(f"pyramid-{feature_out}-relu", nn.ReLU(inplace=True))
            n_input_features = feature_out
            curr_dim = curr_dim // 2

        # latent layer (initial)

        self.latent_feature = nn.Sequential()

        self.latent_feature.add_module(
            name=f'input-{n_input_features}-convt',
            module=nn.ConvTranspose2d(
                latent_size,
                n_input_features,
                kernel_size=4,
                padding=4,
                bias=False
            )
        )

        self.latent_feature.add_module(name=f'input-{n_input_features}-batchnorm', module=nn.BatchNorm2d(n_input_features))
        self.latent_feature.add_module(name=f'input-{n_input_features}-relu', module=nn.ReLU(inplace=True))

        # hidden layers

        self.hidden_layers = nn.Sequential()
        for layer in range(hidden_size):
            self.hidden_layers.add_module(
                name=f"hidden-layers-{layer}-{n_input_features}-conv",
                module=nn.Conv2d(n_input_features, n_input_features, kernel_size=4, padding=1, bias=False),
            )
            self.hidden_layers.add_module(
                name=f"hidden-layers-{layer}-{n_input_features}-batchnorm", module=nn.BatchNorm2d(n_input_features)
            )
            self.hidden_layers.add_module(
                name=f"hidden-layers-{layer}-{n_input_features}-relu", module = nn.ReLU(inplace=True)
            )

        # tanh layer at the end of the decoder network for classification

        self.tanh_layer = nn.Sequential()
        self.tanh_layer.add_module(
            name=f'tanh-{n_input_features}-convt',
            module=nn.ConvTranspose2d(n_input_features, n_input_features, kernel_size=4, padding=1, bias=False)
        )
        self.tanh_layer.add_module(
            name=f'tanh-{n_input_features}-tanh',
            module=nn.Tanh()
        )

    def forward(self, input_tensor: Tensor):
        output = self.latent_feature(input_tensor)
        output = self.pyramid_features(output)
        output = self.hidden_layers(output)
        output = self.tanh_layer(output)
        return output
