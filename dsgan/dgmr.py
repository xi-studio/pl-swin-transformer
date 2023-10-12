import torch
import pytorch_lightning as pl
from common import LatentConditioningStack, ContextConditioningStack
from generators import Sampler, Generator

import sys


class DGMR(pl.LightningModule):
    """Deep Generative Model of Radar"""

    def __init__(
        self,
        forecast_steps: int = 18,
        input_channels: int = 1,
        output_shape: int = 256,
        conv_type: str = "standard",
        latent_channels: int = 768,
        context_channels: int = 384,
        encode_type: str = "allchannel",
    ):
        """
        Nowcasting GAN is an attempt to recreate DeepMind's Skillful Nowcasting GAN from https://arxiv.org/abs/2104.00954
        but slightly modified for multiple satellite channels
        Args:
            forecast_steps: Number of steps to predict in the future
            input_channels: Number of input channels per image
            visualize: Whether to visualize output during training
            gen_lr: Learning rate for the generator
            disc_lr: Learning rate for the discriminators, shared for both temporal and spatial discriminator
            conv_type: Type of 2d convolution to use, see satflow/models/utils.py for options
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
            num_samples: Number of samples of the latent space to sample for training/validation
            grid_lambda: Lambda for the grid regularization loss
            output_shape: Shape of the output predictions, generally should be same as the input shape
            latent_channels: Number of channels that the latent space should be reshaped to,
                input dimension into ConvGRU, also affects the number of channels for other linked inputs/outputs
            pretrained:
        """
        super(DGMR, self).__init__()
        self.forecast_steps = forecast_steps
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.input_channels = input_channels
        self.conditioning_stack = ContextConditioningStack(
            input_channels=input_channels,
            conv_type=conv_type,
            num_context_steps=2,
            output_channels=self.context_channels,
        )
        self.conditioning_stack_15 = ContextConditioningStack(
            input_channels=1,
            conv_type=conv_type,
            num_context_steps=2,
            output_channels=self.context_channels,
        )
        self.conditioning_stack_others = ContextConditioningStack(
            input_channels=8,
            conv_type=conv_type,
            num_context_steps=2,
            output_channels=self.context_channels,
        )
        self.latent_stack = LatentConditioningStack(
            # shape=(8 * self.input_channels, output_shape // 32, output_shape // 32),
            output_channels=self.latent_channels,
        )
        self.encode_type = encode_type
        assert encode_type in ["allchannel", "add", "cat"]

        self.sampler = Sampler(
            forecast_steps=self.forecast_steps,
            latent_channels=self.latent_channels,
            context_channels=self.context_channels,
            output_channels=self.input_channels,
        )
        self.generator = Generator(
            self.conditioning_stack,
            self.conditioning_stack_15,
            self.conditioning_stack_others,
            self.latent_stack,
            self.sampler,
            self.encode_type,
        )

    def forward(self, x):
        x = self.generator(x)
        return x
