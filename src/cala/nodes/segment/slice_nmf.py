from collections.abc import Hashable, Mapping
from typing import Annotated as A
from typing import Any

import numpy as np
import xarray as xr
from noob import Name, process_method
from noob.node import Node
from pydantic import Field

from cala.arrays import AXIS, Buffer, Footprint, Trace
from cala.logging import init_logger
from cala.util import rank1nmf


class SliceNMF(Node):
    min_frames: int
    """Wait until this number of frames to begin detecting."""
    detect_thresh: float
    """Minimum detection threshold for brightness fluctuation."""
    reprod_tol: float
    """Mean pixel value error tolerance for reproduced slice from the new component"""

    nmf_kwargs: dict[str, Any] = Field(default_factory=dict)

    error_: float = Field(None)

    _logger = init_logger(__name__)

    @process_method
    def process(
        self, residuals: Buffer, energy: xr.DataArray, cell_size: int
    ) -> tuple[A[list[Footprint], Name("new_fps")], A[list[Trace], Name("new_trs")]]:

        if residuals.array.sizes[AXIS.frame_dim] < self.min_frames:
            return None, None

        fps = []
        trs = []

        res = residuals.array.copy()

        while np.max(energy) >= self.detect_thresh:
            # Find and analyze neighborhood of maximum variance
            slice_ = self._get_max_energy_slice(arr=res, energy_landscape=energy, radius=cell_size)

            a_new, c_new = self._local_nmf(
                slice_=slice_,
                spatial_sizes={k: v for k, v in res.sizes.items() if k in AXIS.spatial_dims},
            )

            l1_error = self.error_ / np.sum(slice_.values)
            l0_error = self.error_ / np.prod(slice_.shape).astype(float)

            energy.loc[{ax: slice_.coords[ax] for ax in AXIS.spatial_dims}] = 0

            if min(l1_error, l0_error) <= self.reprod_tol:
                fps.append(Footprint.from_array(a_new))
                trs.append(Trace.from_array(c_new))
                res.loc[{ax: slice_.coords[ax] for ax in AXIS.spatial_dims}] = l1_error
            else:
                res.loc[{ax: slice_.coords[ax] for ax in AXIS.spatial_dims}] = l0_error

        return (fps, trs) if fps else (None, None)

    def _get_max_energy_slice(
        self, arr: xr.DataArray, energy_landscape: xr.DataArray, radius: int
    ) -> xr.DataArray:
        """Find neighborhood around point of maximum variance."""
        # Find maximum point
        max_coords = energy_landscape.argmax(dim=AXIS.spatial_dims)

        # Define neighborhood
        window = {
            ax: slice(
                max(0, pos.values - radius),
                min(energy_landscape.sizes[ax], pos.values + radius + 1),
            )
            for ax, pos in max_coords.items()
        }

        # Embed the actual coordinates onto the array
        neighborhood = arr.isel(window).assign_coords(
            {ax: energy_landscape.coords[ax][pos] for ax, pos in window.items()}
        )

        return neighborhood

    def _local_nmf(
        self, slice_: xr.DataArray, spatial_sizes: Mapping[Hashable, int]
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Perform local rank-1 Non-negative Matrix Factorization.

        Uses scikit-learn's NMF implementation to decompose the neighborhood
        into spatial (a) and temporal (c) components.

        Args:
            slice_ (xr.DataArray): Local region of residual buffer.
                Shape: (frames × height × width)

        Returns:
            tuple[xr.DataArray, xr.DataArray]:
                - Spatial component a_new (height × width)
                - Temporal component c_new (frames)
        """
        # Reshape neighborhood to 2D matrix (time × space)
        R = slice_.transpose(AXIS.frame_dim, ...).data.reshape((slice_.sizes[AXIS.frame_dim], -1)).T

        mean_R = np.mean(R, axis=1)
        # nan_mask = np.isnan(mean_R)

        a, c, self.error_ = rank1nmf(R, mean_R)

        # Convert back to xarray with proper dimensions and coordinates
        c_new = xr.DataArray(
            c.squeeze(),
            dims=[AXIS.frame_dim],
            coords=slice_[AXIS.frame_dim].coords,
        )

        # Create full-frame zero array with proper coordinates
        a_new = xr.DataArray(
            np.zeros(tuple(spatial_sizes.values())),
            dims=tuple(spatial_sizes.keys()),
            coords={ax: np.arange(size) for ax, size in spatial_sizes.items()},
        )

        # Place the NMF result in the correct location
        a_new.loc[{ax: slice_.coords[ax] for ax in AXIS.spatial_dims}] = xr.DataArray(
            a.squeeze().reshape(tuple(slice_.sizes[ax] for ax in AXIS.spatial_dims)),
            dims=AXIS.spatial_dims,
            coords={ax: slice_.coords[ax] for ax in AXIS.spatial_dims},
        )

        # normalize against the original video (as in whatever the residual used at the time)
        factor = a_new.max()
        a_new = a_new / factor
        c_new = c_new * factor

        return a_new, c_new
