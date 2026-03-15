from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name, process_method
from pydantic import BaseModel, PrivateAttr
from scipy.sparse import csr_matrix

from cala.arrays import AXIS, Buffer, Frame
from cala.arrays.models import Footprints, Traces


class Residuer(BaseModel):
    _mean: np.ndarray = PrivateAttr(None)
    _sq_mean: np.ndarray = PrivateAttr(None)

    @process_method
    def update(
        self, residuals: Buffer, frame: Frame, footprints: Footprints, traces: Traces
    ) -> tuple[A[Buffer, Name("movie")], A[xr.DataArray, Name("std")]]:
        """
        Computes and maintains a buffer of residual signals.

        This method implements the residual computation by subtracting the
        reconstructed signal from the original data. It maintains only the
        most recent frames as specified by the buffer length.

        The residual buffer contains the recent history of unexplained variance
        in the data after accounting for known components.

        The computation follows the equation:
            R_buf = [Y − [A, b][C; f]][:, t′ − l_b + 1 : t′]
            where:
            - Y is the data matrix (pixels × time)
            - [A, b] is the spatial footprint matrix of neurons and background
            - [C; f] is the temporal traces matrix of neurons and background
            - t' is the current timestep
            - l_b is the buffer length
            - R_buf is the resulting residual buffer

        Args:
            footprints (Footprints): Spatial footprints of all components.
                Shape: (components × height × width)
            traces (Traces): Temporal traces of all components.
                Shape: (components × time)
        """

        if footprints.array is None or traces.array is None:
            if residuals.array is None:
                residuals.array = frame.array.expand_dims(dim=AXIS.frame_dim)
            else:
                residuals.append(frame.array)
            std = self._update_std(frame.array)

            return residuals, std

        Y = frame.array
        C = traces.array.isel({AXIS.frame_dim: -1})  # (components,)
        A = footprints.array
        A_pix = (
            A.transpose(AXIS.component_dim, ...)
            .data.reshape((A.sizes[AXIS.component_dim], -1))
            .tocsr()
        )

        R_curr, flag = _find_overestimates(Y=Y, A=A_pix, C=C)
        if flag:
            C = _align_overestimates(A_pix=A_pix, C_latest=C, R_latest=R_curr)
            traces.array.loc[{AXIS.frame_dim: -1}] = C

        # if recently discovered, set to zero (or a small number). otherwise, just append
        preserve_area = _get_new_estimators_area(A=A, C=C)
        if preserve_area is not None:
            mask = preserve_area.as_numpy()
            residuals.array_ *= mask
            self._mean *= mask
            self._sq_mean *= mask

        # we're not fully modifying for the negative minimum, so we need to clip
        R_curr = _get_residuals(Y=Y, A=A_pix, C=C).clip(min=0)
        residuals.append(R_curr)

        std = self._update_std(R_curr)

        return residuals, std

    def _update_std(self, arr: xr.DataArray) -> xr.DataArray:
        """median: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=614292"""
        eta = 1 / (arr[AXIS.frame_coord].item() + 1)

        if self._mean is None:
            self._mean = arr.values
            self._sq_mean = np.square(arr.values)
        else:
            self._mean += eta * (arr.values - self._mean)
            self._sq_mean += eta * (np.square(arr.values) - self._sq_mean)
        return xr.DataArray(np.sqrt(self._sq_mean - np.square(self._mean)), dims=arr.dims)


def _init_energy(res: xr.DataArray) -> xr.DataArray:
    return res.std(dim=AXIS.frame_dim)


def _get_residuals(Y: xr.DataArray, A: csr_matrix, C: xr.DataArray) -> xr.DataArray:
    return Y - xr.DataArray((C.data @ A).reshape(Y.shape), dims=AXIS.spatial_dims)


def _find_overestimates(
    Y: xr.DataArray, A: xr.DataArray, C: xr.DataArray
) -> tuple[xr.DataArray, bool]:
    R_curr = _get_residuals(Y, A, C)
    return R_curr, R_curr.min() < -np.finfo(np.float32).eps


def _align_overestimates(
    A_pix: csr_matrix, C_latest: xr.DataArray, R_latest: xr.DataArray
) -> xr.DataArray:
    """
    !!We're assuming there's no completely occluded component. This might be a problem eventually!!
    """
    R = R_latest.values
    unlayered_stamp = _find_unlayered_footprints(A_pix)  # same up to here

    R_rel = unlayered_stamp * np.minimum(R, 0).reshape(1, -1)  # same up to here to 2e-6 and nan
    RA = A_pix.power(-1).multiply(R_rel).tocsr()  # same up to here to 2e-6 and neginf
    dC = RA.minimum(0).sum(axis=1)  # .nanmin(axis=1, explicit=True)
    # divide by the number of active pixels to normalize negative (prevents outliers)
    dC_norm = np.asarray(dC).squeeze() / np.array([a.nnz for a in RA])

    return (C_latest + dC_norm).clip(min=0)


def _find_unlayered_footprints(A: csr_matrix) -> np.ndarray:
    coords = A.nonzero()[1]
    pixels, counts = np.unique(coords, return_counts=True)
    mask = np.isin(coords, pixels[counts == 1])
    vals = A.data[mask]
    locs = coords[mask]
    ret = np.zeros(A.shape[1])
    ret[locs] = vals
    return ret


def _get_new_estimators_area(A: xr.DataArray, C: xr.DataArray) -> xr.DataArray | None:
    targets = C[AXIS.detect_coord].values >= (C[AXIS.frame_coord].item() - 1)

    if any(targets):
        idx = np.where(targets)[0]
        nonzeros = A.data.nonzero()
        target_mask = np.isin(nonzeros[0], idx)
        target_coords = tuple(nonzero[target_mask] for nonzero in nonzeros[1:])
        target_area = np.ones(A.shape[1:], dtype=bool)
        target_area[target_coords] = 0
        return xr.DataArray(target_area, dims=A.dims[1:])
    else:
        return None
