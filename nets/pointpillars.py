from typing import List, Tuple

import torch
import torch.nn as nn


class PFNLayer(nn.Module):
    """A single Pillar Feature Net layer.

    Each PFN layer applies a linear transformation to the input point
    features, followed by batch normalisation and GELU activation.  The
    output is then aggregated across the points within a pillar using
    a max operation.  For non-last layers, the aggregated feature is
    repeated for every point and concatenated back onto the per-point
    features so that the next PFN layer has access to both local and
    pillar-level information.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature tensor.
    out_channels : int
        Number of output channels after the linear layer.  For non-last
        layers the output will be concatenated back onto the input, so
        the next layer will see `out_channels * 2` channels.
    last_layer : bool, optional
        If True, indicates that this is the final PFN layer.  In this
        case the aggregated feature is returned directly without
        concatenation.
    """

    def __init__(self, in_channels: int, out_channels: int, last_layer: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_layer = last_layer

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single PFN layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(B, M, P, C_in)`, where `B` is the
            batch size, `M` the number of pillars, `P` the maximum
            number of points per pillar and `C_in` the number of
            channels.
        mask : torch.Tensor
            Boolean tensor of shape `(B, M, P)` indicating which
            positions in `x` correspond to real points (True) and
            which are padding (False).  Masking ensures that padded
            elements do not affect the aggregation.

        Returns
        -------
        torch.Tensor
            If `last_layer` is False, returns an updated tensor of
            shape `(B, M, P, out_channels * 2)`; otherwise returns the
            aggregated pillar features of shape `(B, M, out_channels)`.
        """
        # Apply linear transformation and activation
        B, M, P, _ = x.shape
        x = x.reshape(-1, self.in_channels)  # (B*M*P, C_in)
        x = self.linear(x)
        # BatchNorm expects (N, C), apply across flattened points
        x = self.norm(x)
        x = self.activation(x)
        x = x.reshape(B, M, P, self.out_channels)

        # Mask the padded points by replacing them with negative infinity
        # so that they are ignored during max pooling
        minus_inf = torch.finfo(x.dtype).min
        x_masked = x.masked_fill(~mask.unsqueeze(-1), minus_inf)

        # Aggregate across points with max
        x_max = x_masked.max(dim=2)[0]  # (B, M, C_out)

        if self.last_layer:
            # For the last layer we return aggregated pillar features
            return x_max

        # For intermediate layers, repeat aggregated features to match
        # the number of points and concatenate with point features
        x_max_repeated = x_max.unsqueeze(2).repeat(1, 1, P, 1)
        # Concatenate along the channel dimension
        x_out = torch.cat([x, x_max_repeated], dim=-1)
        return x_out


def get_paddings_indicator(actual_num: torch.Tensor, max_num: int) -> torch.Tensor:
    """Create a boolean mask for valid points in a pillar.

    Parameters
    ----------
    actual_num : torch.Tensor
        Tensor of shape `(B, M)` containing the number of real points
        in each pillar.
    max_num : int
        The maximum number of points per pillar.

    Returns
    -------
    torch.Tensor
        A boolean mask of shape `(B, M, max_num)` where entries are
        True for valid points and False for padding.
    """
    # actual_num: (B, M) -> (B, M, 1)
    actual_num = actual_num.unsqueeze(-1)
    # repeat to (B, M, max_num)
    repeated = actual_num.repeat(1, 1, max_num)
    # arange to create shape (max_num,) -> (1,1,max_num)
    # and compare
    indices = torch.arange(max_num, device=actual_num.device).view(1, 1, -1)
    mask = indices < repeated
    return mask


class PillarFeatureNet(nn.Module):
    """Compute per-pillar feature vectors from raw point clouds.

    This class follows the PointPillars methodology described in the
    original paper.  It takes voxelised point data (with positions and additional features)
    and augments each point with cluster offsets (difference to the mean
    point within the pillar), voxel centre offsets (difference to the
    geometric centre of the pillar) and optionally the Euclidean
    distance of the point from the origin.  The augmented features are
    processed by a sequence of `PFNLayer`s to obtain a single feature
    vector per pillar.

    Parameters
    ----------
    num_input_features : int
        Number of channels in the raw point features. This is typically 4 (x, y, z, intensity).
    feat_channels : list of int
        A list specifying the output channels of each PFN layer.  The
        length of this list determines the number of PFN layers.
    voxel_size : tuple of float
        Physical size of a single voxel along (x, y, z) axes.  This is
        used to compute the voxel centre offsets.
    point_cloud_range : tuple of float
        The spatial range of the point cloud in order (x_min, y_min,
        z_min, x_max, y_max, z_max).  Used to compute voxel centre
        offsets.
    with_cluster_offset : bool, optional
        Whether to include cluster offset features.  Default: True.
    with_voxel_offset : bool, optional
        Whether to include voxel centre offset features.  Default: True.
    with_distance : bool, optional
        Whether to include the Euclidean distance of each point from the
        origin as a feature.  Default: False.
    """

    def __init__(
        self,
        num_input_features: int,
        feat_channels: List[int],
        voxel_size: Tuple[float, float, float],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        with_cluster_offset: bool = True,
        with_voxel_offset: bool = True,
        with_distance: bool = False,
    ) -> None:
        super().__init__()
        self.with_cluster_offset = with_cluster_offset
        self.with_voxel_offset = with_voxel_offset
        self.with_distance = with_distance

        # Precompute voxel centre offsets and scaling factors
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        vox_size_x, vox_size_y, vox_size_z = voxel_size
        self.register_buffer("x_offset", torch.tensor(vox_size_x / 2 + x_min))
        self.register_buffer("y_offset", torch.tensor(vox_size_y / 2 + y_min))
        self.register_buffer("z_offset", torch.tensor(vox_size_z / 2 + z_min))
        self.register_buffer("vox_size_x", torch.tensor(vox_size_x))
        self.register_buffer("vox_size_y", torch.tensor(vox_size_y))
        self.register_buffer("vox_size_z", torch.tensor(vox_size_z))

        # Determine the number of extra channels added by feature decorations
        num_extra_feat = 0
        if self.with_cluster_offset:
            num_extra_feat += 3  # dx, dy, dz
        if self.with_voxel_offset:
            num_extra_feat += 3  # x_centre offset, y_centre offset, z_centre offset
        if self.with_distance:
            num_extra_feat += 1  # Euclidean distance

        in_channels = num_input_features + num_extra_feat

        # Construct PFN layers
        layers: List[nn.Module] = []
        for i, out_channels in enumerate(feat_channels):
            last_layer = i == (len(feat_channels) - 1)
            layers.append(PFNLayer(in_channels, out_channels, last_layer=last_layer))
            # For non-last layers the next input channels double due to
            # concatenation of aggregated features
            if not last_layer:
                in_channels = out_channels * 2
            else:
                in_channels = out_channels
        self.pfn_layers = nn.ModuleList(layers)

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-pillar features from voxelised inputs.

        Parameters
        ----------
        voxel_features : torch.Tensor
            Tensor of shape `(B, M, P, C_in)` containing the raw point
            features for each pillar, where `B` is the batch size,
            `M` the number of non-empty pillars, `P` the maximum number
            of points per pillar and `C_in` the number of input feature
            channels (e.g., 4 for x, y, z, intensity).
        voxel_coords : torch.Tensor
            Tensor of shape `(B, M, 3)` containing the integer voxel
            coordinates `(z, y, x)` for each pillar.
        num_points_per_voxel : torch.Tensor
            Tensor of shape `(B, M)` giving the actual number of points in
            each pillar.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(B, M, C_out)` containing the aggregated
            feature vector for each pillar, where `C_out` is the last
            value in `feat_channels`.
        """
        assert voxel_features.dim() == 4, "voxel_features should be (B, M, P, C_in)"
        B, M, P, C_in = voxel_features.shape

        device = voxel_features.device
        dtype = voxel_features.dtype

        # Compute cluster offsets: difference of each point from the mean
        features = voxel_features
        decorations: List[torch.Tensor] = []
        if self.with_cluster_offset:
            # compute mean per pillar over points: shape (B, M, 1, 3)
            # Use masked mean to ignore padded points
            mask = get_paddings_indicator(num_points_per_voxel, P).to(device)
            points_mean = (features[:, :, :, :3] * mask.unsqueeze(-1)).sum(dim=2, keepdim=True) / num_points_per_voxel.unsqueeze(-1).unsqueeze(-1).clamp(min=1)
            f_cluster = features[:, :, :, :3] - points_mean
            decorations.append(f_cluster)

        if self.with_voxel_offset:
            # compute voxel centre coordinates
            # voxel_coords is (B, M, 3): (z, y, x)
            # convert to x, y, z order and scale to real coordinates
            # x_centre = x_idx * vox_size_x + x_offset
            x_centres = voxel_coords[:, :, 2].to(dtype) * self.vox_size_x + self.x_offset
            y_centres = voxel_coords[:, :, 1].to(dtype) * self.vox_size_y + self.y_offset
            z_centres = voxel_coords[:, :, 0].to(dtype) * self.vox_size_z + self.z_offset
            # reshape to (B, M, 1) then broadcast to (B, M, P, 1)
            x_centres = x_centres.unsqueeze(-1).unsqueeze(-1)
            y_centres = y_centres.unsqueeze(-1).unsqueeze(-1)
            z_centres = z_centres.unsqueeze(-1).unsqueeze(-1)
            f_center = torch.zeros((B, M, P, 3), dtype=dtype, device=device)
            f_center[..., 0] = features[..., 0] - x_centres.squeeze(-1)
            f_center[..., 1] = features[..., 1] - y_centres.squeeze(-1)
            f_center[..., 2] = features[..., 2] - z_centres.squeeze(-1)
            decorations.append(f_center)

        if self.with_distance:
            # Euclidean distance from origin of each point
            distance = torch.norm(features[:, :, :, :3], dim=-1, keepdim=True)
            decorations.append(distance)

        if decorations:
            features = torch.cat([features] + decorations, dim=-1)

        # Mask invalid padded positions
        mask = get_paddings_indicator(num_points_per_voxel, P).to(device)

        # Apply PFN layers
        x = features
        for layer in self.pfn_layers:
            x = layer(x, mask)
        # x is aggregated feature of shape (B, M, C_out)
        return x


class PointPillarsScatter(nn.Module):
    """Scatter per-pillar features to a dense BEV grid.

    Given the feature vectors for each non-empty pillar and their integer
    coordinates `(z, y, x)`, this module constructs a dense tensor of shape
    `(B, C, Z, X)` by taking a channel-wise maximum over all pillars that
    map to the same `(z, x)` location.  This collapses the Y (height)
    dimension and matches the BEV format expected.  The
    scatter operation iterates over the batch and uses index based
    assignment for efficiency.

    Parameters
    ----------
    bev_shape : tuple of int
        The shape of the output BEV grid as `(Z, Y, X)`.  Note that the
        Y dimension is ignored (collapsed) when producing the output.
    """

    def __init__(self, bev_shape: Tuple[int, int, int]) -> None:
        super().__init__()
        self.Z, self.Y, self.X = bev_shape

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor) -> torch.Tensor:
        """Scatter features into a dense tensor.

        Parameters
        ----------
        voxel_features : torch.Tensor
            Per-pillar feature tensor of shape `(B, M, C)`.
        voxel_coords : torch.Tensor
            Integer coordinate tensor of shape `(B, M, 3)` corresponding
            to `(z, y, x)` indices for each pillar.

        Returns
        -------
        torch.Tensor
            Dense BEV feature tensor of shape `(B, C, Z, X)` where Y
            dimension has been collapsed.
        """
        B, M, C = voxel_features.shape
        device = voxel_features.device
        # Initialise canvas (B, C, Z, X) with negative infinity so that
        # max pooling across y works correctly
        canvas = voxel_features.new_full((B, C, self.Z, self.X), float('-inf'))
        # Iterate over batch
        for b in range(B):
            feats = voxel_features[b]  # (M, C)
            coords = voxel_coords[b]   # (M, 3)
            z_idx = coords[:, 0].long()
            x_idx = coords[:, 2].long()
            # For each pillar assign features by taking max over y
            # Use scatter to take elementwise maximum
            # Flatten features for each channel
            for i in range(C):
                # canvas[b,i,z_idx,x_idx] = torch.maximum(canvas[b,i,z_idx,x_idx], feats[:,i])
                current = canvas[b, i, z_idx, x_idx]
                canvas[b, i, z_idx, x_idx] = torch.maximum(current, feats[:, i])
        # Replace -inf with zeros (empty positions)
        canvas[canvas == float('-inf')] = 0
        return canvas


class PointPillarsEncoder(nn.Module):
    """High-level PointPillars encoder

    This class wraps the `PillarFeatureNet` and `PointPillarsScatter` to
    produce BEV features from voxelised point cloud data.  It is
    compatible with the existing VoxelNet interface used in
    `SegnetTransformerLiftFuse` and can be selected by setting
    `lidar_encoder_type` to `'point_pillars'` in the configuration.

    Parameters
    ----------
    latent_dim : int
        Number of channels in the output BEV feature map.  This should
        match the latent dimension used by the segmentation network.
    Z : int
        Number of bins along the Z axis in the memory grid.
    Y : int
        Number of bins along the Y axis in the memory grid.
    X : int
        Number of bins along the X axis in the memory grid.
    point_feature_dim : int, optional
        Number of channels in the raw point feature (e.g., 4 for x, y, z,
        intensity).  Default: 4.
    with_distance : bool, optional
        Whether to include the Euclidean distance as an additional
        feature.  Default: False.
    vox_util : object, optional
        Utility object providing voxelisation parameters.  Must have
        attributes `XMIN`, `YMIN`, `ZMIN`, `XMAX`, `YMAX`, `ZMAX`.  If
        None is provided the user must set voxel_size and
        point_cloud_range manually via other means (not recommended).
    """

    def __init__(
        self,
        latent_dim: int,
        Z: int,
        Y: int,
        X: int,
        point_feature_dim: int = 4,
        with_distance: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.Z = Z
        self.Y = Y
        self.X = X
        self.point_feature_dim = point_feature_dim
        self.with_distance = with_distance

        # FeatureNet build lazily 
        self.feature_net: PillarFeatureNet | None = None
        self._bounds_initialized = False

        # Scatter only based on (Z, Y, X)
        self.scatter = PointPillarsScatter(bev_shape=(Z, Y, X))

    def _build_feature_net_from_vox_util(self, vox_util, device: torch.device) -> None:
        """Initialisiere PillarFeatureNet einmalig aus vox_util."""
        if vox_util is None:
            raise ValueError("PointPillarsEncoder.forward erwartet ein vox_util-Objekt, got None.")

        
        x_min, x_max = float(vox_util.XMIN), float(vox_util.XMAX)
        y_min, y_max = float(vox_util.YMIN), float(vox_util.YMAX)
        z_min, z_max = float(vox_util.ZMIN), float(vox_util.ZMAX)

        # Voxelsize 
        voxel_size_x = (x_max - x_min) / float(self.X)
        voxel_size_y = (y_max - y_min) / float(self.Y)
        voxel_size_z = (z_max - z_min) / float(self.Z)
        voxel_size = (voxel_size_x, voxel_size_y, voxel_size_z)

        point_cloud_range = (x_min, y_min, z_min, x_max, y_max, z_max)

        # FeatureNet
        self.feature_net = PillarFeatureNet(
            num_input_features=self.point_feature_dim,
            feat_channels=[self.latent_dim],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            with_cluster_offset=True,
            with_voxel_offset=True,
            with_distance=self.with_distance,
        ).to(device)

        self._bounds_initialized = True

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        vox_util=None,
        dinovoxel=None,
    ) -> torch.Tensor:
        """Encode voxelised LiDAR features into a BEV map.

        Parameters
        ----------
        voxel_features : torch.Tensor
            Tensor of shape `(B, M, P, point_feature_dim)` containing
            voxelised points and their associated features. The first
            three channels must be XYZ coordinates.
        voxel_coords : torch.Tensor
            Tensor of shape `(B, M, 3)` with integer `(z, y, x)` indices.
        num_points_per_voxel : torch.Tensor
            Tensor of shape `(B, M)` indicating the actual number of
            points in each pillar.
        vox_util : object
            VoxUtil instance providing XMIN/XMAX/... for bounds.
        dinovoxel : unused, kept for interface compatibility.

        Returns
        -------
        torch.Tensor
            BEV feature map of shape `(B, latent_dim, Z, X)`.
        """
        # Lazy-Init of the FeatureNet-Bounds from vox_util
        if not self._bounds_initialized:
            self._build_feature_net_from_vox_util(vox_util, voxel_features.device)

        # Compute per-pillar features (B, M, latent_dim)
        pillar_features = self.feature_net(
            voxel_features=voxel_features,
            voxel_coords=voxel_coords,
            num_points_per_voxel=num_points_per_voxel,
        )

        # Scatter features into dense BEV map (collapse Y dimension)
        bev = self.scatter(pillar_features, voxel_coords)  # (B, latent_dim, Z, X)
        return bev
