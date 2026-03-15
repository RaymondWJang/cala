import cv2
import numpy as np
import pytest
import xarray as xr
from noob import SynchronousRunner, Tube
from noob.node import Node, NodeSpecification

from cala.arrays import AXIS
from cala.nodes.io import stream
from cala.nodes.segment.quality_control import separate_by_filter
from cala.plot import footprint_rims


@pytest.fixture(
    params=[
        "SingleCellSource",
        "TwoCellsSource",
        "SeparateSource",
        "TwoOverlappingSource",
        "GradualOnSource",
        "SplitOffSource",
    ],
    scope="module",
)
def source(request):
    return Node.from_specification(
        NodeSpecification(id="source", type=f"cala.testing.{request.param}")
    )


@pytest.fixture(scope="module")
def tube(tmp_path_factory):
    tube = Tube.from_specification("cala-odl", {"radius": 10})
    # tube.cube.arrays["traces"].params["zarr_path"] = tmp_path_factory.mktemp("traces")

    return tube


@pytest.fixture(scope="module")
def runner(tube, request):
    return SynchronousRunner(tube=tube)


@pytest.fixture(scope="module")
def results(runner, source):
    """
    Let's maybe add handling similar traces? (ex. tanh and exp)

    """
    gen = runner.iter(n=source.instance.n_frames)
    toy = source.instance.toy.model_copy()
    src_name = source.spec.type_.split(".")[-1]
    preprocessed_frames = []
    for fr in gen:
        preprocessed_frames.append(fr["prep"].array)
        fps = runner.tube.state.assets["footprints"].obj
        trs = runner.tube.state.assets["traces"].obj

    return {
        "model": toy,
        "name": src_name,
        "prep_movie": preprocessed_frames,
        "footprints": fps.array,
        "traces": trs.array,
    }


def test_component_counts(results):
    """Pipeline ends up with correct component counts"""

    if results["name"] == "SplitOffSource":
        # one should be deprecated but is not :(
        assert (
            results["traces"].sizes[AXIS.component_dim]
            == results["model"].traces.array.sizes[AXIS.component_dim] + 1
        )
    else:
        assert (
            results["traces"].sizes[AXIS.component_dim]
            == results["model"].traces.array.sizes[AXIS.component_dim]
        )


@pytest.mark.xfail
def test_trace_correlation(results) -> None:
    """
    Components' trace becomes similar to the expected trace as we
    approach the end of analysis.

    """

    tr_corr = xr.corr(
        results["model"].traces.array,
        results["traces"].rename(AXIS.component_rename),
        dim=AXIS.frame_dim,
    )
    for corr in tr_corr:
        assert np.isclose(corr.max(), 1, atol=1e-5)


@pytest.mark.xfail
def test_footprint_similarity(results):
    """
    Components' footprint become similar in size and shape to
    the expected footprints as the approach progresses.

    """
    raise NotImplementedError()


@pytest.mark.xfail
def test_reconstructed_movie(results):
    """
    Reconstructed movie from the shape / trace detection should
    be identical to (or at least close to) the expected movie as the analysis
    progresses.

    """
    raise NotImplementedError()


VIDEOS = [f"long_recording/{i}.avi" for i in range(1)]


def test_recursive():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("circles_x2.mp4", fourcc, 30.0, (1200, 600))

    gen = stream(VIDEOS)
    tube = Tube.from_specification("cala-unraveled", {"cell_size": 10})
    runner = SynchronousRunner(tube)
    max_frame = np.zeros((600, 600))
    A = runner.tube.state.assets["footprints"]

    def event_cb(event) -> None:
        nonlocal max_frame
        nonlocal A
        if event["node_id"] == "preprocess":
            max_frame = np.maximum(max_frame, event["value"].array.values)
            frame_bgr = cv2.cvtColor(
                event["value"].array.values.astype(np.uint8) * 2, cv2.COLOR_GRAY2BGR
            )
            max_bgr = cv2.cvtColor(max_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            if A.obj.array is not None:
                accepted, rejected = separate_by_filter(A.obj.array, 0.8, 24)
                circles = footprint_rims(accepted)
                frame_bgr[:, :, 0][circles] = 255
                max_bgr[:, :, 0][circles] = 255

            out.write(np.concatenate([frame_bgr, max_bgr], axis=1))

    runner.add_callback(event_cb)
    for idx, arr in enumerate(gen):
        runner.process(frame=arr, epoch=idx)

    out.release()
