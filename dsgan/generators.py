import einops
import torch
import torch.nn.functional as F
from torch.nn.modules.pixelshuffle import PixelShuffle

# from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import spectral_norm
from typing import List
from common import GBlock, UpsampleGBlock
from layers import ConvGRU
from layers.utils import get_conv_layer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class Sampler(torch.nn.Module):
    def __init__(
        self,
        forecast_steps: int = 18,
        latent_channels: int = 768,
        context_channels: int = 384,
        output_channels: int = 1,
    ):
        """
        Sampler from the Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        The sampler takes the output from the Latent and Context conditioning stacks and
        creates one stack of ConvGRU layers per future timestep.
        Args:
            forecast_steps: Number of forecast steps
            latent_channels: Number of input channels to the lowest ConvGRU layer
        """
        super().__init__()
        self.forecast_steps = forecast_steps
        output_channels = 1
        self.convGRU1 = ConvGRU(
            input_channels=latent_channels + context_channels,
            output_channels=context_channels * output_channels,
            kernel_size=3,
        )
        self.gru_conv_1x1 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=context_channels * output_channels,
                out_channels=latent_channels,
                kernel_size=(1, 1),
            )
        )
        self.g1 = GBlock(
            input_channels=latent_channels, output_channels=latent_channels, hw=8
        )
        self.up_g1 = UpsampleGBlock(
            input_channels=latent_channels, output_channels=latent_channels // 2, hw=8
        )

        self.convGRU2 = ConvGRU(
            input_channels=latent_channels // 2
            + context_channels * output_channels // 2,
            output_channels=context_channels * output_channels // 2,
            kernel_size=3,
        )
        self.gru_conv_1x1_2 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=context_channels * output_channels // 2,
                out_channels=latent_channels // 2,
                kernel_size=(1, 1),
            )
        )
        self.g2 = GBlock(
            input_channels=latent_channels // 2,
            output_channels=latent_channels // 2,
            hw=16,
        )
        self.up_g2 = UpsampleGBlock(
            input_channels=latent_channels // 2,
            output_channels=latent_channels // 4,
            hw=16,
        )

        self.convGRU3 = ConvGRU(
            input_channels=latent_channels // 4
            + context_channels * output_channels // 4,
            output_channels=context_channels * output_channels // 4,
            kernel_size=3,
        )
        self.gru_conv_1x1_3 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=context_channels * output_channels // 4,
                out_channels=latent_channels // 4,
                kernel_size=(1, 1),
            )
        )
        self.g3 = GBlock(
            input_channels=latent_channels // 4,
            output_channels=latent_channels // 4,
            hw=32,
        )
        self.up_g3 = UpsampleGBlock(
            input_channels=latent_channels // 4,
            output_channels=latent_channels // 8,
            hw=32,
        )

        self.convGRU4 = ConvGRU(
            input_channels=latent_channels // 8
            + context_channels * output_channels // 8,
            output_channels=context_channels * output_channels // 8,
            kernel_size=3,
        )
        self.gru_conv_1x1_4 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=context_channels * output_channels // 8,
                out_channels=latent_channels // 8,
                kernel_size=(1, 1),
            )
        )
        self.g4 = GBlock(
            input_channels=latent_channels // 8,
            output_channels=latent_channels // 8,
            hw=64,
        )
        self.up_g4 = UpsampleGBlock(
            input_channels=latent_channels // 8,
            output_channels=latent_channels // 16,
            hw=64,
        )

        self.bn = torch.nn.BatchNorm2d(latent_channels // 16)
        # self.bn = torch.nn.BatchNorm2d(latent_channels // 16)#, momentum=0.001)
        self.ln = torch.nn.LayerNorm([latent_channels // 16, 128, 128])
        self.relu = torch.nn.ReLU()
        self.conv_1x1 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=latent_channels // 16,
                out_channels=4,  # * output_channels,
                kernel_size=(1, 1),
            )
        )

        self.depth2space = PixelShuffle(upscale_factor=2)

    def forward(
        self, conditioning_states: List[torch.Tensor], latent_dim: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform the sampling from Skillful Nowcasting with GANs
        Args:
            conditioning_states: Outputs from the `ContextConditioningStack` with the 4 input states, ordered from largest to smallest spatially
            latent_dim: Output from `LatentConditioningStack` for input into the ConvGRUs

        Returns:
            forecast_steps-length output of images for future timesteps

        """
        # Iterate through each forecast step
        # Initialize with conditioning state for first one, output for second one
        init_states = conditioning_states
        # Expand latent dim to match batch size
        # latent_dim = einops.repeat(
        #     latent_dim, "b c h w -> (repeat b) c h w", repeat=init_states[0].shape[0]
        # )
        # print('latent_dim', latent_dim.shape)  # B, 320, 8, 8
        hidden_states = [latent_dim] * self.forecast_steps

        # Layer 4 (bottom most)

        hidden_states = self.convGRU1(hidden_states, init_states[3])
        hidden_states = [self.gru_conv_1x1(h) for h in hidden_states]

        hidden_states = [self.g1(h) for h in hidden_states]

        hidden_states = [self.up_g1(h) for h in hidden_states]

        # Layer 3.

        hidden_states = self.convGRU2(hidden_states, init_states[2])
        hidden_states = [self.gru_conv_1x1_2(h) for h in hidden_states]

        hidden_states = [self.g2(h) for h in hidden_states]

        hidden_states = [self.up_g2(h) for h in hidden_states]

        # Layer 2.

        hidden_states = self.convGRU3(hidden_states, init_states[1])
        hidden_states = [self.gru_conv_1x1_3(h) for h in hidden_states]
        hidden_states = [self.g3(h) for h in hidden_states]
        hidden_states = [self.up_g3(h) for h in hidden_states]

        # Layer 1 (top-most).

        hidden_states = self.convGRU4(hidden_states, init_states[0])
        hidden_states = [self.gru_conv_1x1_4(h) for h in hidden_states]
        hidden_states = [self.g4(h) for h in hidden_states]
        hidden_states = [self.up_g4(h) for h in hidden_states]

        # Output layer.
        hidden_states = [F.relu(self.ln(h)) for h in hidden_states]
        # hidden_states = [F.relu(self.bn(h)) for h in hidden_states]
        hidden_states = [F.relu(h) for h in hidden_states]
        # print('before conv1*1', [i.size() for i in hidden_states])
        hidden_states = [self.conv_1x1(h) for h in hidden_states]
        # print('after conv1*1', [i.size() for i in hidden_states])
        hidden_states = [self.depth2space(h) for h in hidden_states]

        # Convert forecasts to a torch Tensor
        forecasts = torch.stack(hidden_states, dim=1)
        # print('forecasts',forecasts.size())
        return forecasts


class Generator(torch.nn.Module):
    def __init__(
        self,
        conditioning_stack: torch.nn.Module,
        conditioning_stack_15: torch.nn.Module,
        conditioning_stack_others: torch.nn.Module,
        latent_stack: torch.nn.Module,
        sampler: torch.nn.Module,
        encode_type: str,
    ):
        """
        Wraps the three parts of the generator for simpler calling
        Args:
            conditioning_stack:
            latent_stack:
            sampler:
        """
        super().__init__()
        self.conditioning_stack = conditioning_stack
        self.conditioning_stack_15 = conditioning_stack_15
        self.conditioning_stack_others = conditioning_stack_others
        self.latent_stack = latent_stack
        self.sampler = sampler
        self.encode_type = encode_type
        if self.encode_type == "cat":
            conv2d = get_conv_layer("standard")

            self.conv_1x1_list = torch.nn.ModuleList(
                [
                    spectral_norm(
                        conv2d(
                            in_channels=j,
                            out_channels=j // 2,
                            kernel_size=1,
                        ),
                        eps=0.0001,
                    )
                    for i, j in zip(range(4), [40, 80, 160, 320])
                ]
            )

    def forward(self, x):
        # print('generator input',x.size())
        if self.encode_type == "allchannel":
            conditioning_states = self.conditioning_stack(x)
        elif self.encode_type == "add":
            conditioning_states_15 = self.conditioning_stack_15(x[:, :, 0:1])
            # print('1.5 size after: ',[i.size() for i in conditioning_states_15])
            conditioning_states_others = self.conditioning_stack_others(
                x[
                    :,
                    :,
                    1:,
                ]
            )
            conditioning_states = conditioning_states_15 + conditioning_states_others
        elif self.encode_type == "cat":
            conditioning_states_15 = self.conditioning_stack_15(x[:, :, 0:1])
            # print('1.5 size after: ',[i.size() for i in conditioning_states_15])
            conditioning_states_others = self.conditioning_stack_others(
                x[
                    :,
                    :,
                    1:,
                ]
            )
            # print('others size after: ', [i.size() for i in conditioning_states_others])
            conditioning_states_cat = [
                torch.cat([a, b], dim=1)
                for a, b in zip(conditioning_states_15, conditioning_states_others)
            ]
            # print('cat after: ', [i.size() for i in conditioning_states])
            conditioning_states = []
            for i in range(4):
                conditioning_states.append(
                    self.conv_1x1_list[i](conditioning_states_cat[i])
                )

        # print('after conditioning ',[i.size() for i in conditioning_states])
        latent_dim = self.latent_stack(x)
        # print('late dim ', latent_dim.size())
        # [mf]
        # latent_dim *= 0
        x = self.sampler(conditioning_states, latent_dim)

        # print( '<0 result fraction:', (x[:,0]<0).sum() /(torch.numel(x[:,0]))*1.0)
        x = torch.nn.ReLU6()(x) / 6
        # print('>0.8 result fraction:', (x[:, 0] > 0.8).sum() / (torch.numel(x[:, 0])) * 1.0)
        return x
