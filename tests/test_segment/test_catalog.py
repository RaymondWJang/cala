import numpy as np
import pytest
import xarray as xr
from noob.node import NodeSpecification

from cala.arrays import AXIS, Buffer, Footprints, Traces
from cala.nodes.segment import Cataloger, SliceNMF
from cala.nodes.segment.catalog import _register
from cala.testing.catalog_depr import CatalogerDepr
from cala.testing.util import expand_boundary, split_footprint


@pytest.fixture(scope="module")
def slice_nmf():
    return SliceNMF.from_specification(
        spec=NodeSpecification(
            id="test_slice_nmf",
            type="cala.nodes.segment.SliceNMF",
            params={"min_frames": 10, "detect_thresh": 1, "reprod_tol": 0.001},
        )
    )


@pytest.fixture(scope="module")
def cataloger():
    return Cataloger.from_specification(
        spec=NodeSpecification(
            id="test",
            type="cala.nodes.segment.Cataloger",
            params={
                "age_limit": 100,
                "trace_smooth_kwargs": {"sigma": 2},
                "shape_smooth_kwargs": {"ksize": (1, 1), "sigmaX": 0},
                "merge_threshold": 0.8,
                "val_threshold": 0.5,
                "cnt_threshold": 5,
            },
        )
    )


@pytest.fixture(scope="module")
def cataloger_depr():
    return CatalogerDepr.from_specification(
        spec=NodeSpecification(
            id="test",
            type="cala.testing.catalog_depr.CatalogerDepr",
            params={
                "age_limit": 100,
                "smooth_kwargs": {"sigma": 2},
                "merge_threshold": 0.8,
                "val_threshold": 0.5,
                "cnt_threshold": 5,
            },
        )
    )


def test_register(single_cell, cataloger):
    """
    Can register a component with its footprint and trace,
    adding appropriate dimensions and coordinates

    """
    footprints = single_cell.footprints.array.drop_vars([AXIS.id_coord, AXIS.detect_coord])
    traces = single_cell.traces.array.drop_vars([AXIS.id_coord, AXIS.detect_coord])
    fp, tr = _register(shapes=footprints, tracks=traces)

    # values do not change
    assert np.array_equal(fp.to_numpy(), footprints.to_numpy())
    assert np.array_equal(tr.to_numpy(), traces.to_numpy())
    # get coordinates
    assert fp[AXIS.id_coord].item() == tr[AXIS.id_coord].item()
    assert (
        fp[AXIS.detect_coord].item()
        == tr[AXIS.detect_coord].item()
        == tr[AXIS.detect_coord].max().item()
    )


@pytest.fixture(
    params=[
        pytest.param(("single_cell", 16), id="single_cell"),
        pytest.param(("two_cells", 16), id="two_cells"),
        pytest.param(("two_connected_cells", 16), id="two_connected_cells"),
        pytest.param(("four_separate_cells", 5), id="four_separate_cells"),
        pytest.param(("four_connected_cells", 5), id="four_connected_cells"),
    ]
)
def chunks_setup(request):
    return {"id": request.param[0], "n_split": request.param[1]}


@pytest.fixture
def chunks(request, chunks_setup):
    toy = request.getfixturevalue(chunks_setup["id"])
    footprints = toy.footprints.array.as_numpy().transpose(AXIS.component_dim, ...)
    traces = toy.traces.array.as_numpy().transpose(AXIS.component_dim, ...)

    shape_chunks = []
    trace_chunks = []

    for shape, trace in zip(footprints, traces):
        shapes = split_footprint(shape, chunks_setup["n_split"])
        border = expand_boundary(shapes > 0)
        shapes += border * 1e-17
        tracks = xr.concat([trace] * len(shapes), dim=AXIS.component_dim)

        shape_chunks.append(shapes)
        trace_chunks.append(tracks)

    return xr.concat(shape_chunks, dim=AXIS.component_dim), xr.concat(
        trace_chunks, dim=AXIS.component_dim
    )


def test_monopartite_merge_matrix(cataloger, cataloger_depr, chunks, chunks_setup):
    """
    Should be able to accurately detect which candidate shapes
    qualify for merge with each other

    """
    merge_matrix = cataloger._monopartite_merge_matrix(*chunks)
    merge_matrix_depr = cataloger_depr._merge_matrix(*chunks)
    assert merge_matrix.equals(merge_matrix_depr)
    # need a better way to do this... check # of diagonal boxes, maybe?


def test_merge_candidates(cataloger, chunks, request, chunks_setup):
    """
    A set of incoming candidates should get properly merged with each other
    Tested by checking recreated movies from merged shape against original movie

    """
    fp_chunks, tr_chunks = chunks
    fp_chunks = fp_chunks.drop_vars([AXIS.id_coord, AXIS.detect_coord])
    tr_chunks = tr_chunks.drop_vars([AXIS.id_coord, AXIS.detect_coord])

    merge_matrix = cataloger._monopartite_merge_matrix(fp_chunks, tr_chunks)
    footprints, traces = cataloger._merge_candidates(fp_chunks, tr_chunks, merge_matrix)

    model = request.getfixturevalue(chunks_setup["id"])
    assert len(footprints) == model.n_components

    movie = model.make_movie().array
    xr.testing.assert_allclose(movie, traces @ footprints)


def test_bipartite_merge_groups(cataloger, chunks, request, chunks_setup):
    """
    Should be able to accurately detect which candidate shapes
    qualify to get absorbed into a known existing component

    """
    model = request.getfixturevalue(chunks_setup["id"])
    footprints = (
        model.footprints.array.transpose(AXIS.component_dim, ...)
        .data.reshape((model.footprints.array.sizes[AXIS.component_dim], -1))
        .tocsr()
    )
    traces = model.traces.array.as_numpy().transpose(AXIS.component_dim, ...)
    fp_chunks, tr_chunks = chunks
    merge_groups = cataloger._bipartite_merge_groups(fp_chunks, tr_chunks, footprints, traces)

    expect_grp, expect_cnt = np.unique(fp_chunks[AXIS.id_coord].values, return_counts=True)
    result_grp, result_cnt = np.unique(merge_groups, return_counts=True)

    if expect_cnt.size == 1:
        assert result_cnt.size == 1
    else:
        assert np.array_equal(expect_cnt, result_cnt[result_grp > 0])


def test_absorb_component(slice_nmf, cataloger, single_cell):
    """
    TODO: This test is too weak and have been missing edge cases:
        1. When two new components are merging into a single component (m x n merging)

    """
    buff = Buffer(size=100)
    buff.array = single_cell.make_movie().array
    new_component = slice_nmf.process(buff, energy=buff.array.std(dim=AXIS.frame_dim), cell_size=10)

    A = single_cell.footprints.array

    new_fp, new_tr = new_component
    fp, tr = cataloger._absorb_component(
        new_fp[0].array.expand_dims(dim="component"),
        new_tr[0].array.expand_dims(dim="component"),
        A.data.reshape((A.shape[0], -1)).tocsr(),
        single_cell.traces.array,
        [0],
    )

    movie_result = (fp @ tr).as_numpy()

    movie_new_comp = new_fp[0].array @ new_tr[0].array
    movie_expected = (single_cell.make_movie().array + movie_new_comp).transpose(*movie_result.dims)

    xr.testing.assert_allclose(movie_result, movie_expected)


def test_process(cataloger, chunks, chunks_setup, request):
    model = request.getfixturevalue(chunks_setup["id"])
    footprints, traces = model.footprints, model.traces
    fp_chunks, tr_chunks = chunks
    fp_chunks = [Footprints.from_array(fp_chunks.drop_vars([AXIS.id_coord, AXIS.detect_coord]))]
    tr_chunks = [Traces.from_array(tr_chunks.drop_vars([AXIS.id_coord, AXIS.detect_coord]))]

    new_fps, new_trs = cataloger.process(fp_chunks, tr_chunks, footprints, traces)

    result = new_trs.array @ new_fps.array
    # result is double the original movie, since the incoming chunks have been
    # merged with the existing.
    expected = traces.array @ footprints.array * 2
    xr.testing.assert_allclose(result.as_numpy(), expected.as_numpy())
