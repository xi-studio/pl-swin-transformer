import torch
from torch.nn.modules.pixelshuffle import PixelUnshuffle

# from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F
from skillful_nowcasting.common import DBlock


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        num_spatial_frames: int = 8,
        conv_type: str = "standard",
        mask: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.spatial_discriminator = SpatialDiscriminator(
            input_channels=self.input_channels,
            num_timesteps=num_spatial_frames,
            conv_type=conv_type,
        )

        self.temporal_discriminator = TemporalDiscriminator(
            input_channels=self.input_channels, conv_type=conv_type
        )
        self.mask = mask
        if self.mask:
            self.masked_discriminator = MaskDiscriminator(
                input_channels=input_channels,
                num_timesteps=num_spatial_frames,
                conv_type=conv_type,
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.mask:
            masked_loss = self.masked_discriminator(y, x[:, 4:])
        spatial_loss = self.spatial_discriminator(x)  # [2b,22,1,h,w] =>

        temporal_loss = self.temporal_discriminator(x)

        if self.mask:
            # print( 'spatial: ',spatial_loss.detach().cpu().numpy().squeeze(), '\n',
            #        'temporal: ',temporal_loss.detach().cpu().numpy().squeeze(), '\n',
            #        'masked: ', masked_loss.detach().cpu().numpy().squeeze())
            return torch.cat([spatial_loss, temporal_loss, masked_loss], dim=1)
        else:
            # print( 'spatial: ',spatial_loss.detach().cpu().numpy().squeeze(), '\n',
            #        'temporal: ',temporal_loss.detach().cpu().numpy().squeeze(), '\n',)
            return torch.cat([spatial_loss, temporal_loss], dim=1)


class TemporalDiscriminator(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        num_layers: int = 3,
        conv_type: str = "standard",
    ):
        """
        Temporal Discriminator from the Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of channels per timestep
            crop_size: Size of the crop, in the paper half the width of the input images
            num_layers: Number of intermediate DBlock layers to use
            conv_type: Type of 2d convolutions to use, see satflow/models/utils.py for options
        """
        super().__init__()
        self.downsample = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_chn = 48
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=internal_chn * input_channels,
            conv_type="3d",
            first_relu=False,
        )
        self.d2 = DBlock(
            input_channels=internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            conv_type="3d",
        )
        self.intermediate_dblocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )

        self.d_last = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        self.fc = spectral_norm(torch.nn.Linear(2 * internal_chn * input_channels, 1))
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(2 * internal_chn * input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.downsample(x)
        # [mf] random crop to 128*128 not pooling
        h_start = torch.randint(0, 128, (1,))
        w_start = torch.randint(0, 128, (1,))
        x = x[:, :, :, h_start : h_start + 128, w_start : w_start + 128]

        x = self.space2depth(x)
        # Have to move time and channels
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        # 2 residual 3D blocks to halve resolution if image, double number of channels and reduce
        # number of time steps
        x = self.d1(x)
        x = self.d2(x)
        # Convert back to T x C x H x W
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        # Per Timestep part now, same as spatial discriminator
        representations = []
        for idx in range(x.size(1)):
            # Intermediate DBlocks
            # Three residual D Blocks to halve the resolution of the image and double
            # the number of channels.
            rep = x[:, idx, :, :, :]
            for d in self.intermediate_dblocks:
                rep = d(rep)
            # One more D Block without downsampling or increase number of channels
            rep = self.d_last(rep)

            rep = torch.sum(F.relu(rep), dim=[2, 3])
            rep = self.bn(rep)
            rep = self.fc(rep)

            # rep = self.fc(rep)
            representations.append(rep)
        # The representations are summed together before the ReLU
        x = torch.stack(representations, dim=1)
        # Should be [Batch, N, 1]
        x = torch.sum(x, keepdim=True, dim=1)
        return x


class SpatialDiscriminator(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        num_timesteps: int = 8,
        num_layers: int = 4,
        conv_type: str = "standard",
    ):
        """
        Spatial discriminator from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of input channels per timestep
            num_timesteps: Number of timesteps to use, in the paper 8/18 timesteps were chosen
            num_layers: Number of intermediate DBlock layers to use
            conv_type: Type of 2d convolutions to use, see satflow/models/utils.py for options
        """
        super().__init__()
        # Randomly, uniformly, select 8 timesteps to do this on from the input
        self.input_channels = input_channels
        self.num_timesteps = num_timesteps
        # First step is mean pooling 2x2 to reduce input by half
        self.mean_pool = torch.nn.AvgPool2d(2)
        self.space2depth = PixelUnshuffle(
            downscale_factor=2
        )  # frames = tf.nn.space_to_depth(frames, block_size=2)
        internal_chn = 24
        self.d1 = DBlock(
            input_channels=4 * self.input_channels,
            output_channels=2 * internal_chn * self.input_channels,
            first_relu=False,
            conv_type=conv_type,
        )
        self.intermediate_dblocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * self.input_channels,
                    output_channels=2 * internal_chn * self.input_channels,
                    conv_type=conv_type,
                )
            )
        self.d6 = DBlock(
            input_channels=2 * internal_chn * self.input_channels,
            output_channels=2 * internal_chn * self.input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        # Spectrally normalized linear layer for binary classification
        self.fc = spectral_norm(
            torch.nn.Linear(2 * internal_chn * self.input_channels, 1)
        )
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(2 * internal_chn * self.input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should be the chosen 8 or so
        idxs = torch.randint(low=0, high=x.size()[1], size=(self.num_timesteps,))
        representations = []
        """
        [mf] 153-154 from discrimninator.txt
        # Process each of the n inputs independently.
        frames = tf.reshape(frames, [b * n, h, w, c])
        """

        for idx in idxs:
            rep = self.mean_pool(x[:, idx, :, :, :])  # 128x128

            rep = self.space2depth(rep)  # 64x64x4
            rep = self.d1(rep)  # 32x32
            # Intermediate DBlocks
            for d in self.intermediate_dblocks:
                rep = d(rep)
            rep = self.d6(rep)  # 2x2
            rep = torch.sum(F.relu(rep), dim=[2, 3])
            rep = self.bn(rep)
            rep = self.fc(rep)
            """
            Pseudocode from DeepMind
            # Sum-pool the representations and feed to spectrally normalized lin. layer.
            y = tf.reduce_sum(tf.nn.relu(y), axis=[1, 2])
            y = layers.BatchNorm(calc_sigma=False)(y)
            output_layer = layers.Linear(output_size=1)
            output = output_layer(y)

            # Take the sum across the t samples. Note: we apply the ReLU to
            # (1 - score_real) and (1 + score_generated) in the loss.
            output = tf.reshape(output, [b, n, 1])
            output = tf.reduce_sum(output, keepdims=True, axis=1)
            return output
            """
            representations.append(rep)

        # The representations are summed together before the ReLU
        x = torch.stack(representations, dim=1)
        # Should be [Batch, N, 1]
        x = torch.sum(x, keepdim=True, dim=1)
        return x


class MaskDiscriminator(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        num_timesteps: int = 8,
        num_layers: int = 4,
        conv_type: str = "standard",
    ):
        """
        Spatial discriminator from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of input channels per timestep
            num_timesteps: Number of timesteps to use, in the paper 8/18 timesteps were chosen
            num_layers: Number of intermediate DBlock layers to use
            conv_type: Type of 2d convolutions to use, see satflow/models/utils.py for options
        """
        super().__init__()
        # Randomly, uniformly, select 8 timesteps to do this on from the input
        self.num_timesteps = num_timesteps
        # First step is mean pooling 2x2 to reduce input by half
        self.mean_pool = torch.nn.AvgPool2d(2)
        self.space2depth = PixelUnshuffle(
            downscale_factor=2
        )  # frames = tf.nn.space_to_depth(frames, block_size=2)
        internal_chn = 24
        self.d1 = DBlock(
            input_channels=4 * input_channels * 2,
            output_channels=2 * internal_chn * input_channels,
            first_relu=False,
            conv_type=conv_type,
        )
        self.intermediate_dblocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )
        self.d6 = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        # Spectrally normalized linear layer for binary classification
        self.fc = spectral_norm(torch.nn.Linear(2 * internal_chn * input_channels, 1))
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(2 * internal_chn * input_channels)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x should be the chosen 8 or so
        idxs = torch.randint(low=0, high=y.size()[1], size=(self.num_timesteps,))
        representations = []
        """
        [mf] 153-154 from discrimninator.txt
        # Process each of the n inputs independently.
        frames = tf.reshape(frames, [b * n, h, w, c])
        """

        for idx in idxs:
            print(x.size(), y.size())  # [b*2,18,:]
            rep_x = self.mean_pool(x[:, idx, :, :, :])  # 128x128 [2b,0,1,h,w]
            rep_y = self.mean_pool(y[:, idx, :, :, :])  # [2b,0,1,h,w]
            rep = torch.cat([rep_x, rep_y], dim=1)  # [2b,2,h,w]

            rep = self.space2depth(rep)  # 64x64x4
            rep = self.d1(rep)  # 32x32
            # Intermediate DBlocks
            for d in self.intermediate_dblocks:
                rep = d(rep)
            rep = self.d6(rep)  # 2x2
            rep = torch.sum(F.relu(rep), dim=[2, 3])
            rep = self.bn(rep)
            rep = self.fc(rep)
            """
            Pseudocode from DeepMind
            # Sum-pool the representations and feed to spectrally normalized lin. layer.
            y = tf.reduce_sum(tf.nn.relu(y), axis=[1, 2])
            y = layers.BatchNorm(calc_sigma=False)(y)
            output_layer = layers.Linear(output_size=1)
            output = output_layer(y)

            # Take the sum across the t samples. Note: we apply the ReLU to
            # (1 - score_real) and (1 + score_generated) in the loss.
            output = tf.reshape(output, [b, n, 1])
            output = tf.reduce_sum(output, keepdims=True, axis=1)
            return output
            """
            representations.append(rep)

        # The representations are summed together before the ReLU
        res = torch.stack(representations, dim=1)
        # Should be [Batch, N, 1]
        res = torch.sum(res, keepdim=True, dim=1)
        return res
