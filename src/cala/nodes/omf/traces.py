from logging import Logger
from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name, process_method
from pydantic import BaseModel
from scipy.sparse.csgraph import connected_components

from cala.arrays import AXIS, Frame, PopSnap
from cala.arrays.models import Footprints, Overlaps, Traces
from cala.logging import init_logger
from cala.util import norm, stack_sparse


class Tracer(BaseModel):
    tol: float
    max_iter: int

    _logger: Logger = init_logger(__name__)

    @process_method
    def ingest_frame(
        self, traces: Traces, footprints: Footprints, frame: Frame, overlaps: Overlaps
    ) -> tuple[A[Traces, Name("traces")], A[PopSnap, Name("new_fit")]]:
        """
        Update temporal traces using current spatial footprints and frame data.

        This method implements the iterative block coordinate descent update of temporal
        traces with guaranteed convergence under non-negativity constraints. It processes
        components based on their overlap relationships, ensuring that overlapping components
        are updated together for proper convergence.

        Follows the iterative formula:

            c[G_i] = max(c[G_i] + (u[G_i] - V[G_i,:]c)/v[G_i], 0)

        where:
            - c is the temporal traces vector
            - G_i represents component groups
            - u = Ã^T y (projection of current frame)
            - V = Ã^T Ã (gram matrix of spatial components)
            - v = diag{V} (diagonal elements for normalization)

        Args:
            footprints (Footprints): Spatial footprints of all components.
                Shape: (components × height × width)
            frame (xr.DataArray): Current frame data.
                Shape: (height × width)
            overlaps (Overlaps): Sparse matrix indicating component overlaps.
                Shape: (components × components), where entry (i,j) is 1 if
                components i and j overlap, and 0 otherwise.
        """
        if footprints.array is None:
            return traces, PopSnap()

        # Prepare inputs for the update algorithm
        A = stack_sparse(footprints.array, AXIS.component_dim).tocsr().T
        y = frame.array.data.reshape((-1,))
        c = traces.array.isel({AXIS.frame_dim: -1}).copy()

        AtA = (A.T @ A).toarray()

        _, labels = connected_components(
            csgraph=overlaps.array.data, directed=False, return_labels=True
        )
        clusters = [np.where(labels == label)[0] for label in np.unique(labels)]
        C, noisyC = _update_traces(
            y, A, c.values, AtA, iters=self.max_iter, tol=self.tol, groups=clusters
        )
        updated_traces = xr.DataArray(C, dims=c.dims, coords=c.coords).assign_coords(
            {
                AXIS.frame_coord: frame.array[AXIS.frame_coord],
                AXIS.timestamp_coord: frame.array[AXIS.timestamp_coord],
            }
        )

        if traces.zarr_path:
            updated_tr = updated_traces.volumize.dim_with_coords(
                dim=AXIS.frame_dim, coords=[AXIS.frame_coord, AXIS.timestamp_coord]
            )
            traces.append(updated_tr, dim=AXIS.frame_dim)
        else:
            traces.append(updated_traces, dim=AXIS.frame_dim)

        return traces, PopSnap.from_array(updated_traces)


def _update_traces(
    y: np.ndarray,
    A: np.ndarray,
    noisyC: np.ndarray,
    AtA: np.ndarray,
    iters: int = 5,
    tol: float = 1e-3,
    groups: list[list[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve C = argmin_C ||Yr-AC|| using block-coordinate decent
    Parameters
    ----------
    y : array of float, shape (pixels,)
        flattened array of raw data frame
    A : sparse matrix of float, shape (pixels, comps)
        neural shapes
    noisyC : ndarray of float, shape (comps,)
        Initial value of fluorescence intensities.
    AtA : ndarray of float (comps, comps)
        Overlap matrix of shapes A.
    iters : int, optional
        Maximal number of iterations.
    tol : float, optional
        Tolerance.
    groups: list of lists
        groups of components to update in parallel
    """
    AtY = A.T.dot(y)
    num_iters = 0
    C_old = np.zeros_like(noisyC)
    C = noisyC.copy()

    while (norm(C_old - C) >= tol * norm(C_old)) and (num_iters < iters):
        C_old[:] = C
        if groups is None:
            for m in range(len(AtY)):
                noisyC[m] = C[m] + (AtY[m] - AtA[m].dot(C)) / (AtA[m, m] + np.finfo(C.dtype).eps)
                C[m] = max(noisyC[m], 0)
        else:
            for m in groups:
                noisyC[m] = C[m] + (AtY[m] - AtA[m].dot(C)) / (
                    AtA.diagonal()[m] + np.finfo(C.dtype).eps
                )
                C[m] = np.maximum(noisyC[m], 0)
        num_iters += 1

    # noisyC is just C with negative values (unclipped)
    return C, noisyC


def ingest_component(traces: Traces, new_traces: Traces) -> Traces:
    """

    :param traces:
    :param new_traces: Can be either a newly registered trace or an updated existing one.
    """

    c_new = new_traces.array

    if c_new is None:
        return traces

    if traces.array is None:
        traces.array = c_new
        return traces

    total_frames = traces.sizes[AXIS.frame_dim]
    new_n_frames = c_new.sizes[AXIS.frame_dim]

    merged_ids = c_new.attrs.get("replaces")
    if merged_ids:
        mask = np.isin(traces.array[AXIS.id_coord].values, merged_ids)
        traces.deprecate(mask)

    c_pad = _pad_history(c_new, total_frames, np.nan) if total_frames > new_n_frames else c_new

    traces.append(c_pad, dim=AXIS.component_dim)

    return traces


def _pad_history(traces: xr.DataArray, total_nframes: int, value: float = np.nan) -> xr.DataArray:
    """
    Pad unknown historical epochs with values...

    """
    new_nframes = traces.sizes[AXIS.frame_dim]

    c_new = xr.DataArray(
        np.full((traces.sizes[AXIS.component_dim], total_nframes), value),
        dims=[AXIS.component_dim, AXIS.frame_dim],
        coords=traces[AXIS.component_dim].coords,
    )

    c_new.loc[{AXIS.frame_dim: slice(total_nframes - new_nframes, None)}] = traces

    return c_new
