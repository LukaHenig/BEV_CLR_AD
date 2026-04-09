"""
VoxelNeXt Decoder Module
========================

This module defines a simple decoder that can upsample the coarse Bird's-Eye-View
(BEV) feature maps produced by a VoxelNeXt backbone to match a desired BEV
resolution.  The decoder mirrors the spirit of the upsampling layers used in
traditional VoxelNet implementations but remains flexible: the number of
upsampling stages is inferred from the desired output BEVMap size and the assumed
downsampling factor of the backbone.

Example
-------

Given a VoxelNeXt backbone that produces features with spatial dimensions
``(H_in, W_in)`` and a target BEV map of size ``(H_out, W_out)``, this
decoder upsamples the input feature map using a sequence of transposed
convolutions with stride 2 until the height and width match the target
resolution.  A final 1x1 convolution projects the feature channels to the
desired latent dimension.

If the required upsampling ratio is not a power of two (e.g. the target size
is not an integer multiple of the input size), the decoder falls back to
bilinear interpolation to reach the exact resolution.

Parameters
----------
in_channels : int
    Number of channels in the input feature map (from the VoxelNeXt backbone).
latent_dim : int
    Number of channels in the output BEV map.  This should match the latent
    dimension expected by the downstream network (e.g. 128).
target_z : int
    Desired spatial size along the first spatial dimension of the BEV map (Z
    dimension in the segmentation pipeline).  Typical values are 200 or 400.
target_x : int
    Desired spatial size along the second spatial dimension of the BEV map (X
    dimension).  Typical values are the same as ``target_z``.
downsample_factor : int, optional
    Downsampling factor of the VoxelNeXt backbone relative to the desired BEV
    resolution.  The default of 8 corresponds to a backbone that outputs
    1/8-resolution BEV features (e.g. 25x25 from a 200x200 grid).  Adjust this
    value if your backbone uses a different downsampling rate.
"""

#improvment idears: maybe add skip connections, or spatial attention modules, residual blocks, normalization 

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelNeXtDecoder(nn.Module):
    """Decoder that upsamples coarse VoxelNeXt BEV features to a dense BEV map.

    The decoder computes the required number of upsampling stages based on
    ``target_z``, ``target_x`` and ``downsample_factor``.  Each stage uses a
    transposed convolution with stride 2 to double the spatial resolution.
    A final 1x1 convolution projects the feature channels to ``latent_dim``.
    If the output dimensions do not match exactly after the upsampling stages
    (e.g. because the desired resolution is not a power-of-two multiple of the
    input), the decoder performs a bilinear interpolation to reach the exact
    target size.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature map.
    latent_dim : int
        Desired number of output channels.
    target_z : int
        Target height of the BEV map (Z dimension).
    target_x : int
        Target width of the BEV map (X dimension).
    downsample_factor : int, optional
        Factor by which the backbone downsampled the BEV plane.  Must be a
        positive integer.  Defaults to 8.
    """

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 target_z: int,
                 target_x: int,
                 downsample_factor: int = 8) -> None:
        super().__init__()

        if downsample_factor <= 0:
            raise ValueError(
                f"downsample_factor must be a positive integer, got {downsample_factor}"
            )

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.target_z = target_z
        self.target_x = target_x
        self.downsample_factor = downsample_factor

        # Compute the expected coarse feature dimensions produced by VoxelNeXt.
        # We assume the downsampling is the same along both spatial dimensions.
        # If the downsampling factors along Z and X differ, adjust this logic
        # accordingly.
        self.coarse_z = max(1, target_z // downsample_factor)
        self.coarse_x = max(1, target_x // downsample_factor)

        # Determine the number of 2x upsampling stages needed to approach
        # the target resolution.  We keep track of how much upsampling we
        # perform along each dimension separately.
        self.upsample_stages_z = int(math.log2(max(1, downsample_factor)))
        self.upsample_stages_x = int(math.log2(max(1, downsample_factor)))

        # Build upsampling layers.  Each layer upsamples by a factor of 2.
        # We use stride=2 transposed convolutions with kernel size 3 and
        # appropriate padding to double the resolution.  InstanceNorm is used
        # instead of BatchNorm to avoid batch‑size dependencies.
        layers = []
        current_channels = in_channels
        for i in range(max(self.upsample_stages_z, self.upsample_stages_x)):
            stride_h = 2 if i < self.upsample_stages_z else 1
            stride_w = 2 if i < self.upsample_stages_x else 1
            # When stride is 1, kernel_size=1 ensures no spatial change.
            kernel_size_h = 2 if stride_h > 1 else 1
            kernel_size_w = 2 if stride_w > 1 else 1
            padding_h = 0 if stride_h == 1 else 0
            padding_w = 0 if stride_w == 1 else 0
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        current_channels,
                        current_channels,
                        kernel_size=(kernel_size_h, kernel_size_w),
                        stride=(stride_h, stride_w),
                        padding=(padding_h, padding_w),
                        bias=False,
                    ),
                    nn.InstanceNorm2d(current_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.upsample_layers = nn.ModuleList(layers)

        # Final projection to latent_dim channels.  If the number of channels
        # hasn't changed, this becomes an identity.
        if current_channels != latent_dim:
            self.proj = nn.Sequential(
                nn.Conv2d(current_channels, latent_dim, kernel_size=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample a coarse BEV feature map to the target BEV resolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C_in, H_in, W_in)``.  Typically, this is
            the output of the VoxelNeXt backbone where ``H_in`` and ``W_in``
            correspond to ``target_z // downsample_factor`` and
            ``target_x // downsample_factor``.

        Returns
        -------
        torch.Tensor
            Upsampled feature map of shape ``(B, latent_dim, target_z, target_x)``.
        """
        # Apply the upsampling layers.
        for layer in self.upsample_layers:
            x = layer(x)

        # If the output shape does not match the desired resolution exactly,
        # interpolate to the target size.  This handles cases where the
        # downsampling factor is not an exact power of two or where the target
        # dimensions are not divisible by the downsampling factor.
        if x.shape[2] != self.target_z or x.shape[3] != self.target_x:
            x = F.interpolate(
                x, size=(self.target_z, self.target_x), mode="bilinear", align_corners=False
            )

        # Project to the desired latent dimension.
        x = self.proj(x)
        return x