from typing import Annotated as A
from typing import Any

import numpy as np
from noob import Name

from cala.arrays import AXIS
from cala.arrays.models import CompStats, Footprints, Overlaps, PixStats, Traces
from cala.nodes.segment.quality_control import morphology_filter


def deprecate(
    footprints: Footprints,
    traces: Traces,
    pix_stats: PixStats,
    comp_stats: CompStats,
    overlaps: Overlaps,
    mask: np.ndarray[Any, np.dtype[np.bool]] | None = None,
) -> tuple[
    A[Footprints, Name("footprints")],
    A[Traces, Name("traces")],
    A[PixStats, Name("pix_stats")],
    A[CompStats, Name("comp_stats")],
    A[Overlaps, Name("overlaps")],
]:
    """
    Deprecate a set of components from all models.
    """
    # ! Refactor the below two lines to a separate node
    if footprints.array is None:
        return footprints, traces, pix_stats, comp_stats, overlaps
    fps = [
        fp
        for fp in footprints.array.data.reshape(
            (footprints.array.sizes[AXIS.component_dim], -1)
        ).tocsc()
    ]
    passed = morphology_filter(footprints=fps, value_threshold=0.8, count_threshold=20)
    failed = ~np.array(passed)

    if any(failed):
        traces.deprecate(failed, inplace=True)
        footprints.deprecate(failed, inplace=True)
        pix_stats.deprecate(failed, inplace=True)
        comp_stats.deprecate(failed, inplace=True)
        overlaps.deprecate(failed, inplace=True)

    return footprints, traces, pix_stats, comp_stats, overlaps


def find_inactive() -> list[str]:
    """
    Deprecate inactive components
    Component is deemed inactive if its own brightness contribution across
    all of its footprint is below threshold.

    1. has been some time since discovery
    2. within its own footprint, its brightness contribution is lower than
        some % of the minimum of the total brightness contributions from all components?
        - but what if the component is completely occluded sometimes?
    """
    raise NotImplementedError
