from typing import Annotated as A

from noob import Name

from cala.arrays import Movie
from cala.arrays.models import CompStats, Footprints, Overlaps, PixStats, Traces
from cala.nodes.omf.component_stats import ingest_component as update_component_stats
from cala.nodes.omf.footprints import ingest_component as update_footprints
from cala.nodes.omf.overlap import ingest_component as update_overlap
from cala.nodes.omf.pixel_stats import ingest_component as update_pixel_stats
from cala.nodes.omf.traces import ingest_component as update_traces


def update_assets(
    new_footprints: Footprints,
    new_traces: Traces,
    footprints: Footprints,
    traces: Traces,
    pixel_stats: PixStats,
    component_stats: CompStats,
    overlaps: Overlaps,
    buffer: Movie,
) -> tuple[
    A[Traces, Name("traces")],
    A[Footprints, Name("footprints")],
    A[PixStats, Name("pixel_stats")],
    A[CompStats, Name("component_stats")],
    A[Overlaps, Name("overlaps")],
]:
    updated_overlaps = update_overlap(
        overlaps=overlaps, footprints=footprints, new_footprints=new_footprints
    )
    updated_pixel_stats = update_pixel_stats(
        pixel_stats=pixel_stats, frames=buffer, new_traces=new_traces, new_footprints=new_footprints
    )
    updated_component_stats = update_component_stats(
        component_stats=component_stats, traces=traces, new_traces=new_traces
    )
    # Shapes and Traces must be updated last to ensure the arrays do not include
    # the new components before we update the others.
    updated_shapes = update_footprints(footprints=footprints, new_footprints=new_footprints)
    updated_traces = update_traces(traces=traces, new_traces=new_traces)

    return (
        updated_traces,
        updated_shapes,
        updated_pixel_stats,
        updated_component_stats,
        updated_overlaps,
    )
