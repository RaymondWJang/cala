from typing import Annotated as A, Sequence

from noob import Name

from cala.arrays import CompStats, Footprints, Overlaps, PixStats, Traces


def deprecate_except(
    footprints: Footprints,
    traces: Traces,
    pix_stats: PixStats,
    comp_stats: CompStats,
    overlaps: Overlaps,
    keep_mask: Sequence[bool],
) -> tuple[
    A[Footprints, Name("footprints")],
    A[Traces, Name("traces")],
    A[PixStats, Name("pix_stats")],
    A[CompStats, Name("comp_stats")],
    A[Overlaps, Name("overlaps")],
]:
    """
    Deprecate a set of components from all assets.
    """
    traces.deprecate_except(keep_mask)
    # the line below compiles numba. gotta do it like in footprints.ingest_component
    # but then i need to redundantly convert COO -> csr -> COO -> csr -> COO
    footprints.array = footprints.array[keep_mask]
    pix_stats.array = pix_stats.array[keep_mask]
    comp_stats.array = comp_stats.array[keep_mask].T[keep_mask]
    overlaps.array = overlaps.array[keep_mask].T[keep_mask]

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
