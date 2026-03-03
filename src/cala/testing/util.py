from collections.abc import Callable, Iterable
from copy import deepcopy
from itertools import compress
from typing import Any, TypeVar

import cv2
import numpy as np
import xarray as xr

from cala.arrays import AXIS

_TArray = TypeVar("_TArray", xr.DataArray, np.ndarray)


def assert_scalar_multiple_arrays(a: _TArray, b: _TArray, /, rtol: float = 1e-5) -> None:
    """
    Using the Pythagorean Theorem
    Only works with 1-D arrays. (see np.squeeze)
    a: (n, )
    b: (n, )
    """

    if not 0 <= rtol <= 1:
        raise ValueError(f"rtol must be between 0 and 1, got {rtol}.")

    if isinstance(a, np.ndarray):
        assert (
            len(a.shape) == len(b.shape) == 1
        ), f"Arrays must be 1-D. Given: {a.shape=}, {b.shape=}"

    abab = ((a @ b) ** 2).item()
    aabb = (a.dot(a) * b.dot(b)).item()

    assert abab > aabb * (1 - rtol), f"Threshold not met: {abab=} > {aabb * (1 - rtol)=}"


def generate_text_image(
    text: str,
    frame_dims: tuple[int, int] = (256, 256),
    org: tuple[int, int] = None,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font_scale: int = 1,
) -> np.ndarray:
    image = np.zeros(frame_dims, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if org is None:
        org = (frame_dims[0] // 2, frame_dims[1] // 2)

    return cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def shift_by(img: np.ndarray, right_pix: float, down_pix: float) -> np.ndarray:
    M = np.float32([[1, 0, right_pix], [0, 1, down_pix]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def split_2d(
    array_2d: xr.DataArray, n_chunks: int
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Split a 2d array into chunks and returns a list of 2d arrays of the same size.

    n_chunks is the number of chunks to split the array into for each axis

    """
    chunks = []
    coords = []
    hcoords = np.split(array_2d[AXIS.width_coord].values, n_chunks)
    vcoords = np.split(array_2d[AXIS.height_coord].values, n_chunks)
    hchunks = np.hsplit(array_2d.values, n_chunks)
    mask = [hchunk.max() > 0 for hchunk in hchunks]
    for hchunk, hcoord in zip(compress(hchunks, mask), compress(hcoords, mask)):
        vchunks = np.vsplit(hchunk, n_chunks)
        mask = [vchunk.max() > 0 for vchunk in vchunks]
        chunks += list(compress(vchunks, mask))
        coords += [
            {AXIS.width_coord: hcoord, AXIS.height_coord: vcoord}
            for vcoord in compress(vcoords, mask)
        ]

    return chunks, coords


def split_footprint(shape: xr.DataArray, n_chunks: int) -> xr.DataArray:
    """
    Split footprints into square chunks and return a DataArray of them,
    concatenated by components.

    """
    results = []
    chunks, coords = split_2d(shape, n_chunks)
    for chunk, coord in zip(chunks, coords):
        template = xr.DataArray(np.zeros_like(shape), dims=shape.dims, coords=shape.coords)
        template[coord] = chunk
        results.append(template)
    return xr.concat(results, dim=AXIS.component_dim)


def expand_boundary(footprints: xr.DataArray) -> xr.DataArray:
    """
    Perform a binary rectangular dilation for each footprint.

    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return xr.apply_ufunc(
        lambda x: cv2.morphologyEx(x, cv2.MORPH_DILATE, kernel, iterations=1),
        footprints.astype(np.uint8),
        input_core_dims=[AXIS.spatial_dims],
        output_core_dims=[AXIS.spatial_dims],
        vectorize=True,
        dask="parallelized",
    )


def total_gradient_magnitude(image: np.ndarray) -> float:
    """
    Calculates the total gradient magnitude c(I) = || |nabla I| ||_F for a 2D array I.

    This function computes the Frobenius norm of the pixel-wise gradient magnitude map,
    which is a common measure of total image variation or signal energy.

    Args:
        image (np.ndarray): The 2D input array (e.g., an image).

    Returns:
        float: The Frobenius norm of the gradient magnitude map.
    """
    if image.ndim != 2:
        raise ValueError("Input array I must be 2-dimensional.")

    grad_y, grad_x = np.gradient(image)
    grad_magnitude_map = np.sqrt(grad_x**2 + grad_y**2)
    c_I = np.linalg.norm(grad_magnitude_map, ord="fro")

    return c_I


def circle_points(
    img: np.ndarray, locs: Iterable[tuple[int, int]], grayscale: bool = True, **kwargs: Any
) -> np.ndarray:
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if grayscale else deepcopy(img)
    for loc in locs:
        color = cv2.circle(color, loc, **kwargs)
    return color


def check_shift_validity(
    source: xr.DataArray, target: xr.DataArray, shift: np.ndarray, threshold: float, func: Callable
) -> bool:
    expected = np.array([5, 5])
    tester = shift_by(source.values, *expected)
    total = func(tester, target.values)
    result = total - shift
    error = np.linalg.norm(result - expected)
    return error < threshold
