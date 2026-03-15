from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name
from scipy.sparse import csr_matrix

from cala.arrays import AXIS, Buffer, Footprints, Frame, Movie, PixStats, PopSnap, Traces


def ingest_frame(
    pixel_stats: PixStats,
    frame: Frame,
    new_traces: PopSnap,
    footprints: Footprints,
    noise_threshold: float = 1.0,
    trace_threshold: float = 0.0,
) -> PixStats:
    """
    Update pixel statistics using current frame and component.

    This method implements the update equations for pixel-component wise
    statistics matrices. The updates incorporate the current frame and
    temporal component with appropriate time-based scaling.

        W_t = ((t-1)/t)W_{t-1} + (1/t)y_t c_t^T

        where:
        - W_t is the pixel-wise sufficient statistics at time t
        - y_t is the current frame
        - c_t is the current temporal component
        - t is the current timestep

    Args:
        frame (Frame): Current frame y_t.
            Shape: (height × width)
        new_traces (PopSnap): Current temporal component c_t.
            Shape: (components)
    """
    y_t = frame.array
    W = pixel_stats.array
    c_t = new_traces.array
    A = footprints.array

    if c_t is None:
        return pixel_stats

    # Compute scaling factors
    frame_idx = y_t[AXIS.frame_coord].item()
    prev_scale = frame_idx / (frame_idx + 1)
    new_scale = 1 / (frame_idx + 1)

    # Update pixel-component statistics W_t
    # W_t = ((t-1)/t)W_{t-1} + (1/t)y_t c_t^T
    # We only access the footprint area, so we can drastically reduce the calc
    y_flat = y_t.data.flatten()
    # y_flat[y_flat < y_flat.mean() + y_flat.std() * noise_threshold] = 0
    n_components = A.sizes[AXIS.component_dim]
    A_sparse = A.data.reshape((n_components, -1)).tocsr()
    W_flat = W.data.reshape((n_components, -1)) * prev_scale
    mask_coords = A_sparse.nonzero()

    for i in range(n_components):
        if c_t.values[i] > trace_threshold:
            idx = np.where(mask_coords[0] == i)[0]
            coords = mask_coords[1][idx]
            data = y_flat[coords] * c_t.values[i] * new_scale
            # target_masked = csr_matrix((data, coords), shape=y_flat.shape)
            target_masked = np.zeros(y_flat.shape)
            target_masked[coords] = data
            W_flat[i] += target_masked

    CY = W_flat.reshape(W.shape)

    pixel_stats.array = xr.DataArray(CY, dims=W.dims, coords=W.coords)

    return pixel_stats


def ingest_component(
    pixel_stats: PixStats, frames: Movie, new_traces: Traces, new_footprints: Footprints
) -> PixStats:
    """Update pixel statistics with new components.

    Updates W_t according to the equation:

        W_t = [W_t, (1/t)Y_buf c_new^T]

    where:
        t is the current frame index.

    Args:
        frames (Movie): Stack of frames up to current timestep.
        new_traces (Traces): Newly detected components' traces

    Returns:
        PixelStater: Updated pixel statistics matrix
    """
    a_new = new_footprints.array
    c_new = new_traces.array
    W = pixel_stats.array
    Y = frames.array

    if c_new is None:
        return pixel_stats

    if W is None:
        pixel_stats.array = initialize(c_new, Y, a_new)
        return pixel_stats

    # Compute scaling factor (1/t)
    frame_idx = c_new[AXIS.frame_coord].max().item()
    scale = 1 / (frame_idx + 1)

    # Compute outer product of frame and new traces
    # (1/t)Y_buf c_new^T
    CY = outer_with_sparse_mask(masks=a_new, target=Y, right=c_new, scalar=scale)
    w_new = xr.DataArray(CY, dims=a_new.dims, coords=a_new.coords)

    merged_ids = c_new.attrs.get("replaces")
    if merged_ids:
        intact_mask = ~np.isin(W[AXIS.id_coord].values, merged_ids)
        W = W[intact_mask]

    pixel_stats.array = xr.concat([W, w_new], dim=AXIS.component_dim)

    return pixel_stats


def fill_buffer(buffer: Buffer, frame: Frame) -> A[Buffer, Name("buffer")]:
    if buffer.array is None:
        buffer.array = frame.array.volumize.dim_with_coords(
            dim=AXIS.frame_dim, coords=[AXIS.timestamp_coord]
        )
        return buffer

    buffer.append(frame.array)
    return buffer


def initialize(
    traces: xr.DataArray, frames: xr.DataArray, footprints: xr.DataArray
) -> xr.DataArray:
    """
    This transformer calculates the correlation between each pixel's temporal trace
    and each component's temporal activity. The computation provides a measure of
    how well each pixel's activity aligns with each component.

    The computation follows the equation:

        W = Y[:, 1:t']C^T/t'

        where:
        - Y is the data matrix (pixels × time)
        - C is the temporal components matrix (components × time)
        - t' is the current timestep
        - W is the resulting pixel statistics (pixels × components)

    The result W represents the temporal correlation between each pixel
    and each component, normalized by the number of timepoints.

    Args:
        traces (Traces): Temporal traces of all detected fluorescent components.
            Shape: (components × time)
        frames (Movie): Stack of frames up to current timestep.
            Shape: (frames × height × width)
    """
    t_prime = frames[AXIS.frame_coord].max().item() + 1
    CY = outer_with_sparse_mask(masks=footprints, target=frames, right=traces, scalar=1 / t_prime)

    return xr.DataArray(CY, dims=footprints.dims, coords=footprints.coords)


def outer_with_sparse_mask(
    masks: xr.DataArray, target: xr.DataArray, right: xr.DataArray, scalar: int = None
) -> np.ndarray:
    n_frames = target.sizes[AXIS.frame_dim]
    target_flat = target.data.reshape((n_frames, -1))
    n_components = masks.sizes[AXIS.component_dim]
    A_sparse = masks.data.reshape((n_components, -1)).tocsr()
    mask_coords = A_sparse.nonzero()

    cy = []
    for i in range(n_components):
        idx = np.where(mask_coords[0] == i)[0]
        coords = (np.repeat(np.arange(n_frames), len(idx)), np.tile(mask_coords[1][idx], n_frames))
        data = np.concatenate([fr[mask_coords[1][idx]] for fr in target_flat]) * scalar
        target_masked = csr_matrix((data, coords), shape=target_flat.shape)
        cy.append((target_masked.T @ right.values[i]).reshape(masks.shape[1:]))

    return np.stack(cy)
