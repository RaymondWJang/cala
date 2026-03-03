import numpy as np
import pytest
import xarray as xr
from noob import SynchronousRunner, Tube
from noob.node import Node, NodeSpecification

from cala.arrays import AXIS
from cala.nodes.io import stream


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
def tube(source, tmp_path_factory):
    tube = Tube.from_specification("cala-odl")
    tube.nodes["source"] = source
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
    # elif src_name in ["SingleCellSource", "TwoCellsSource", "SeparateSource"]:
    # expected = xr.concat(preprocessed_frames, dim=AXIS.frames_dim)
    # result = (fps.array @ trs.array).transpose(*expected.dims)
    #
    # xr.testing.assert_allclose(expected, result.as_numpy(), atol=1e-5, rtol=1e-5)
    #
    # elif src_name == "SplitOffSource":
    # expected = xr.concat(preprocessed_frames, dim=AXIS.frames_dim)
    # result = (fps.array @ trs.array).transpose(*expected.dims)
    # raise NotImplementedError("Deprecation not implemented")


VIDEOS = [
    "long_recording/0.avi",
    "long_recording/1.avi",
    "long_recording/2.avi",
    "long_recording/3.avi",
    "long_recording/4.avi",
    "long_recording/5.avi",
    "long_recording/6.avi",
    "long_recording/7.avi",
    "long_recording/8.avi",
    "long_recording/9.avi",
]


def test_recursive():
    gen = stream(VIDEOS[:3])
    tube = Tube.from_specification("cala-unraveled", {"cell_size": 8})
    runner = SynchronousRunner(tube)
    for idx, arr in enumerate(gen):
        res = runner.process(frame=arr, epoch=idx)
