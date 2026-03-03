from collections.abc import Callable, Sequence
from pathlib import Path
from shutil import rmtree
from typing import Any
from uuid import uuid4

import numpy as np
import xarray as xr
from noob.event import MetaSignal
from numpydantic.ndarray import NDArray
from sparse import COO
from xarray import Coordinates

from cala.arrays import AXIS


def create_id() -> str:
    return uuid4().hex


def combine_attr_replaces(attrs: Sequence[dict[str, list[str]]], context: None = None) -> dict:
    repl = {item for attr in attrs for item in attr.get("replaces", [])}
    return {"replaces": list(repl)} if repl else {}


def clear_dir(directory: Path | str) -> None:
    for path in Path(directory).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)


def sp_matmul(
    left: xr.DataArray, dim: str, rename_map: dict, right: xr.DataArray | None = None
) -> xr.DataArray:
    """
    Faster than xarray @ (for sparse arrays). The syntax is complicated enough that I'm making a
    utility function

    :param left:
    :param dim:
    :param rename_map:
    :param right:
    """

    ll = stack_sparse(left, dim).tocsr()

    if right is None:
        right = left
        rr = ll
    else:
        rr = stack_sparse(right, dim).tocsr()

    val = ll @ rr.T

    return xr.DataArray(
        COO.from_scipy_sparse(val), dims=[dim, AXIS.duplicate(dim)], coords=left[dim].coords
    ).assign_coords(right[dim].rename(rename_map).coords)


def stack_sparse(da: xr.DataArray, dim: str) -> NDArray:
    return da.transpose(dim, ...).data.reshape((da.sizes[dim], -1))


def concat_components(
    footprints: list[xr.DataArray],
    traces: list[xr.DataArray],
    coords: list[str] = None,
    combine_attrs: Callable | str = "override",
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    A convenience function for concatenating footprints and traces
    simultaneously by components.

    """
    coords = [] if coords is None else coords

    footprints = xr.concat(
        footprints, dim=AXIS.component_dim, coords=coords, combine_attrs=combine_attrs
    )
    traces = xr.concat(traces, dim=AXIS.component_dim, coords=coords, combine_attrs=combine_attrs)

    return footprints, traces


def rank1nmf(
    Ypx: np.ndarray, ain: np.ndarray, iters: int = 10
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    perform a fast rank 1 NMF

    Ypx: (pixels, frames)
    ain: (pixels)
    iters: valid only by period of 4 (seems like i mod 4 = 2 gives good results.
        mod 4 = 3 is marginally better.)

    """
    eps = np.finfo(np.float32).eps
    for t in range(iters):
        cin_res = ain.dot(Ypx)
        cin = np.maximum(cin_res, 0)
        ain = np.maximum(Ypx.dot(cin), 0)
        if t in (0, iters - 1):
            ain /= np.sqrt(ain.dot(ain)) + eps
        elif t % 2 == 0:
            ain /= ain.dot(ain) + eps
    cin_res = ain.dot(Ypx)
    cin = np.maximum(cin_res, 0)
    error = np.linalg.norm(Ypx - np.outer(ain, cin), "fro")
    return ain, cin, error


def norm(c: np.ndarray) -> float:
    """Faster than np.linalg.norm"""
    return np.sqrt(c.ravel().dot(c.ravel()))


def concatenate_coordinates(left: Coordinates, right: Coordinates) -> dict:
    ll = {k: v.values for k, v in left.items()}
    rr = {k: v.values for k, v in right.items()}

    combined = {k: np.concatenate([ll[k], rr[k]]) for k in ll}
    return combined


def capture_subgraph_noevent(value: Any) -> Any:
    """
    A temporary WRONG implementation of capturing NoEvent from TubeNodes.
    Needs to be fixed at the noob level: https://github.com/miniscope/noob/issues/162

    """
    if value is None:
        return MetaSignal.NoEvent
    else:
        return value
