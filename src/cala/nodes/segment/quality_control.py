import numpy as np
import xarray as xr
from sparse import COO


def morphology_filter(
    footprints: list[xr.DataArray], value_threshold: float, count_threshold: int
) -> list[bool]:
    if isinstance(footprints[0].data, COO):
        footprints = [fp.data for fp in footprints]
    return [_sig_pixel_count(fp, value_threshold, count_threshold) for fp in footprints]


def _sig_pixel_count(footprint: xr.DataArray, value_threshold: float, count_threshold: int) -> bool:
    """
    True if the number of pixels with "significant" values (compared to max)
    is above threshold
    """
    return (
        np.sum(footprint.data / footprint.data.max() > value_threshold) > count_threshold
        if footprint.data.size > 0
        else False
    )


def separate_by_filter(
    asset: xr.DataArray, value_threshold: float, count_threshold: int
) -> tuple[list[xr.DataArray], list[xr.DataArray]]:
    A = asset.as_numpy().transpose("component", ...)

    accepted_A = []
    rejected_A = []
    for fp in A:
        if _sig_pixel_count(fp, value_threshold, count_threshold):
            accepted_A.append(fp)
        else:
            rejected_A.append(fp)

    return accepted_A, rejected_A
