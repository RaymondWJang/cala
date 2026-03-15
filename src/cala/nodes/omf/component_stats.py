import numpy as np
import xarray as xr

from cala.arrays import AXIS, Frame, PopSnap
from cala.arrays.models import CompStats, Traces


def ingest_frame(
    component_stats: CompStats, frame: Frame, new_traces: PopSnap, trace_threshold: float = 0.0
) -> CompStats:
    """
    Update component statistics using current frame and component.

        M_t = ((t-1)/t)M_{t-1} + (1/t)c_t c_t^T

        where:
        - M_t is the component-wise sufficient statistics at time t
        - y_t is the current frame
        - c_t is the current temporal component
        - t is the current timestep

    Args:
        frame (Frame): Current frame y_t.
            Shape: (height × width)
        new_traces (Traces): Current temporal component c_t.
            Shape: (components)
    Return:
        component_stats (ComponentStats): Updated component-statistics.
    """
    if new_traces.array is None:
        return component_stats

    # Compute scaling factors
    frame_idx = frame.array[AXIS.frame_coord].item()
    prev_scale = frame_idx / (frame_idx + 1)
    new_scale = 1 / (frame_idx + 1)

    # New frame traces
    c_t = new_traces.array
    c_t[c_t < trace_threshold] = 0

    # Update component-wise statistics M_t
    # M_t = ((t-1)/t)M_{t-1} + (1/t)c_t c_t^T
    new_corr = c_t @ c_t.rename(AXIS.component_rename)
    component_stats.array = (
        prev_scale * component_stats.array + new_scale * new_corr
    ).reset_coords([AXIS.timestamp_coord, AXIS.frame_coord], drop=True)

    return component_stats


def ingest_component(component_stats: CompStats, traces: Traces, new_traces: Traces) -> CompStats:
    """
    Update component statistics with new components.

    Updates M_t according to the equation:
    M_t = (1/t) [ tM_t,         C_buf^T c_new  ]
             [ c_new C_buf^T, ||c_new||^2   ]
    where:
    - t is the current frame index
    - M_t is the existing component statistics
    - C_buf are the traces in the buffer
    - c_new are the new component traces

    Args:
        traces (Traces): Current temporal traces in buffer
        new_traces (Traces): Newly detected temporal components
    """
    c_new = new_traces.array
    c_buf = traces.array
    M = component_stats.array

    no_new = c_new is None
    if no_new:
        return component_stats

    no_M = M is None
    if no_M:
        component_stats.array = initialize(c_new)
        return component_stats

    merged_ids = c_new.attrs.get("replaces", [])
    intact_mask = ~np.isin(M[AXIS.id_coord].values, merged_ids)

    if merged_ids:
        c_buf = c_buf[intact_mask]
        M = M[intact_mask].T[intact_mask]  # symmetric matrix

    # Get current frame index (starting with 1)
    t = c_new[AXIS.frame_coord].max().item() + 1

    # Compute cross-correlation between buffer and new components
    # C_buf^T c_new
    # C_buf has to be the same number of frames as c_new
    bottom_left_corr = c_buf @ c_new.rename(AXIS.component_rename) / t
    top_right_corr = c_buf.rename(AXIS.component_rename) @ c_new / t

    # Compute auto-correlation of new components
    # ||c_new||^2
    auto_corr = c_new @ c_new.rename(AXIS.component_rename) / t

    # Create the block matrix structure
    # Top block: [M_scaled, cross_corr]
    top_block = xr.concat([M, top_right_corr], dim=AXIS.component_dim)
    # Bottom block: [cross_corr.T, auto_corr]
    bottom_block = xr.concat([bottom_left_corr, auto_corr], dim=AXIS.component_dim)
    # Combine blocks
    component_stats.array = xr.concat(
        [top_block, bottom_block], dim=AXIS.duplicate(AXIS.component_dim)
    )

    return component_stats


def initialize(traces: xr.DataArray) -> xr.DataArray:
    """
    calculates the correlation matrix between temporal components
    using their activity traces. The correlation is computed as a normalized
    outer product of the temporal components.

        The computation follows the equation:  M = C @ C.T / t'
        where:
        - C is the temporal components matrix (components × time)
        - t' is the current timestep
        - M is the resulting correlation matrix (components × components)
    """
    # Get temporal components C
    # components x time

    frame_idx = traces[AXIS.frame_coord].max().item()

    # Compute M = C * C.T / t'
    return traces @ traces.rename(AXIS.component_rename) / (frame_idx + 1)
