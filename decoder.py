import math
import torch
from torch import Tensor, nn
from typing import Tuple


class Decoder(nn.Module):
    """Decoder Network. Considering an input vector, tries to decode and generate the original image.
    Uses transposed convolutional layers to reach the image dimension.

    Args:
        output_size (tuple[int, int]): Size of the output image
        latent_size (int): Size of the latent vector z (input of the neural network)
        n_input_channels (int): Number of input channels in the image (greyscale = 1, RGB = 3, CMYK = 4)
        n_features (int): Number of features per convolution layer
        hidden_size (int): Number of hidden convolutional layers. Default is 0.
    """

    def __init__(self,
                 output_size: Tuple[int, int],
                 latent_size: int,
                 n_input_channels: int,
                 n_features: int,
                 hidden_size: int = 0
                 ):
        super().__init__()

        self.latent_input = nn.Sequential()


    def forward(self):
        pass
