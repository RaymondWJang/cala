"""
Since arrays without zarr operations is very straightforward,
this test file is mainly focused on testing zarr-integrated arrays,
with the exception of Buffer.

"""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from cala.arrays import AXIS, Buffer, Traces


@pytest.mark.parametrize("peek_size", [30, 49, 50, 51, 70])
def test_array_assignment(tmp_path, four_connected_cells, peek_size):
    """
    1. array should get assigned to memory if smaller than or equal to peek_size
    2. array should get split to memory and zarr_path at peek_size if
        array is larger than peek_size

    """
    traces = four_connected_cells.traces.array
    n_frames = traces.sizes[AXIS.frame_dim]  # 50 frames

    zarr_traces = Traces(
        zarr_path=tmp_path, peek_size=peek_size, flush_interval=max(1000, peek_size)
    )
    zarr_traces.array = traces
    assert zarr_traces.array_.sizes[AXIS.frame_dim] == min(n_frames, peek_size)
    assert zarr_traces.load_zarr().sizes[AXIS.frame_dim] == max(0, n_frames - peek_size)


@pytest.mark.parametrize("peek_size", [30, 50, 70])
def test_array_peek(tmp_path, four_connected_cells, peek_size):
    """
    .array property returns correctly for peek_size smaller, equal, larger
    than the saved array.

    """
    traces = four_connected_cells.traces.array
    n_frames = traces.sizes[AXIS.frame_dim]  # 50 frames

    zarr_traces = Traces(
        zarr_path=tmp_path, peek_size=peek_size, flush_interval=max(1000, peek_size)
    )
    zarr_traces.array = traces

    assert zarr_traces.array.sizes[AXIS.frame_dim] == min(peek_size, n_frames)


@pytest.mark.parametrize("peek_size", [30, 50, 70])
def test_flush_zarr(four_connected_cells, tmp_path, peek_size):
    """
    _flush_zarr method flushes epochs older than peek_size to zarr_path,
    and leaves only the newest epochs in memory.

    """
    traces = four_connected_cells.traces.array
    n_frames = traces.sizes[AXIS.frame_dim]  # 50 frames

    zarr_traces = Traces(
        zarr_path=tmp_path, peek_size=peek_size, flush_interval=max(1000, peek_size)
    )

    # need to initialize zarr file first to append with _flush_zarr
    zarr_traces.array = traces[:, :0]
    zarr_traces.array_ = traces  # add 50 frames

    zarr_traces._flush_zarr()
    # only peek_size left in memory
    assert zarr_traces.array_.sizes[AXIS.frame_dim] == min(n_frames, peek_size)
    # the rest is in zarr
    assert zarr_traces.load_zarr().sizes[AXIS.frame_dim] == max(0, n_frames - peek_size)


@pytest.mark.parametrize("peek_size, flush_interval", [(30, 70)])
def test_zarr_append_frame(four_connected_cells, tmp_path, peek_size, flush_interval):
    """
    Test that when in-memory array size hits flush_interval, old epochs
        get flushed.

    """
    traces = four_connected_cells.traces.array
    n_frames = traces.sizes[AXIS.frame_dim]  # 50 frames

    zarr_traces = Traces(zarr_path=tmp_path, peek_size=peek_size, flush_interval=flush_interval)
    zarr_traces.array = traces[:, :0]  # just initializing zarr

    # array smaller than flush_interval. does not flush.
    zarr_traces.append(traces, dim=AXIS.frame_dim)
    assert zarr_traces.array_.sizes[AXIS.frame_dim] == n_frames

    # array larger than flush_interval. flushes down to peek_size.
    zarr_traces.append(traces, dim=AXIS.frame_dim)
    assert zarr_traces.array_.sizes[AXIS.frame_dim] == peek_size


@pytest.mark.parametrize("flush_interval", [30])
def test_zarr_append_component(four_connected_cells, tmp_path, flush_interval):
    """
    Test that when adding components,
        (a) it gets appropriately divided between in-memory and zarr arrays.
        (b) in-memory and zarr arrays can be concatenated together afterward.

    """
    traces = four_connected_cells.traces.array

    zarr_traces = Traces(zarr_path=tmp_path, peek_size=20, flush_interval=flush_interval)
    zarr_traces.array = traces[:-1, :]  # forgot the last component! Also, got flushed.
    zarr_traces.append(traces[-1:, :], dim=AXIS.component_dim)  # appendee needs to be 2D

    assert zarr_traces.array_[AXIS.component_dim].equals(traces[AXIS.component_dim])
    assert zarr_traces.load_zarr()[AXIS.component_dim].equals(traces[AXIS.component_dim])
    result = xr.concat([zarr_traces.load_zarr(), zarr_traces.array_], dim=AXIS.frame_dim).compute()
    assert result.equals(traces)


@pytest.mark.parametrize("flush_interval", [30])
def test_flush_after_deprecated(four_connected_cells, tmp_path, flush_interval) -> None:
    traces = four_connected_cells.traces.array
    peek_size = 20
    zarr_traces = Traces(zarr_path=tmp_path, peek_size=peek_size, flush_interval=flush_interval)
    zarr_traces.array = traces

    merged_ids = zarr_traces.array[AXIS.id_coord].values[0]
    intact_mask = ~np.isin(zarr_traces.array[AXIS.id_coord].values, merged_ids)
    zarr_traces.deprecate_except(intact_mask)
    zarr_traces.append(traces[intact_mask], dim=AXIS.frame_dim)

    assert zarr_traces.full_array().equals(xr.concat([traces] * 2, dim=AXIS.frame_dim)[intact_mask])


def test_from_array(four_connected_cells):
    """
    .from_array method can correctly reproduce the array with .array

    """
    traces = four_connected_cells.traces.array
    zarr_traces = Traces.from_array(traces)
    assert zarr_traces.array.equals(traces)


@pytest.mark.parametrize("peek_size", [30, 50, 70])
def test_sizes(four_connected_cells, tmp_path, peek_size):
    """
    The sizes property of the asset combines the sizes
    of the in-memory array and the zarr array.

    """
    traces = four_connected_cells.traces.array

    zarr_traces = Traces(
        zarr_path=tmp_path, peek_size=peek_size, flush_interval=max(1000, peek_size)
    )
    zarr_traces.array = traces

    assert zarr_traces.sizes == traces.sizes


@pytest.mark.xfail
def test_overwrite(four_connected_cells, four_separate_cells):
    """
    test that zarr array can get overwritten.

    """


# two cases of init:
# 1. brick by brick
# 2. lump dump

# two cases of update
# 1. append
# 2. lump update


def test_buffer_assign(four_connected_cells):
    movie = four_connected_cells.make_movie().array
    buff = Buffer(size=10)
    buff.array = movie.isel({AXIS.frame_dim: -1})
    assert buff.array.equals(movie.isel({AXIS.frame_dim: [-1]}))

    buff.array = movie.isel({AXIS.frame_dim: slice(-5, None)})
    assert buff.array.equals(movie.isel({AXIS.frame_dim: slice(-5, None)}))

    buff.array = movie.isel({AXIS.frame_dim: slice(-10, None)})
    assert buff.array.equals(movie.isel({AXIS.frame_dim: slice(-10, None)}))

    buff.array = movie.isel({AXIS.frame_dim: slice(-15, None)})
    assert buff.array.equals(movie.isel({AXIS.frame_dim: slice(-10, None)}))


def test_buffer_append(four_connected_cells):
    movie = four_connected_cells.make_movie().array
    buff = Buffer(size=10)
    buff.array = movie.isel({AXIS.frame_dim: 0})
    buff.append(movie.isel({AXIS.frame_dim: 1}))
    assert buff.array.equals(movie.isel({AXIS.frame_dim: slice(0, 2)}))

    buff.array = movie.isel({AXIS.frame_dim: slice(None, 9)})
    buff.append(movie.isel({AXIS.frame_dim: 9}))
    assert buff.array.equals(movie.isel({AXIS.frame_dim: slice(0, 10)}))
    buff.append(movie.isel({AXIS.frame_dim: 10}))
    assert buff.array.equals(movie.isel({AXIS.frame_dim: slice(1, 11)}))


def test_buffer_speed(single_cell):
    movie = single_cell.make_movie().array
    movie = xr.concat([movie, movie], dim=AXIS.frame_dim)
    buff = Buffer(size=100)
    buff.array = movie

    start = datetime.now()
    iter = 100
    for _ in range(iter):
        buff.append(movie.isel({AXIS.frame_dim: 0}))
        _ = buff.array
    result = (datetime.now() - start) / iter

    start = datetime.now()
    for _ in range(iter):
        xr.concat(
            [movie.isel({AXIS.frame_dim: slice(1, None)}), movie.isel({AXIS.frame_dim: 0})],
            dim=AXIS.frame_dim,
        )
    expected = (datetime.now() - start) / iter

    assert result < expected
