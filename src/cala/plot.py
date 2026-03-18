import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, Generator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def erode_to_average(
    array: np.ndarray,
    average: Literal["mean", "quantile"] = "quantile",
    low: float = None,
    high: float = None,
    mask: np.ndarray = None,
    q: float = None,
) -> np.ndarray:
    """
    return an array where only the mean values (+_ width / 2) are present.
    if mask, only include masked area in the mean calculation / result
    """
    if average == "mean":
        center = array[array > 0].mean() if mask is not None else array.mean()

    elif average == "quantile":
        center = (
            np.quantile(array[array > 0], q).astype(float)
            if mask is not None
            else np.quantile(array, q).astype(float)
        )
    else:
        raise NotImplementedError()

    arr = copy.deepcopy(array)
    if low:
        arr[arr < (center - low / 2.0)] = 0
    if high:
        arr[arr > (center + high / 2.0)] = 0
    return arr


def mark_boundary(image: np.ndarray, kernel_size: int, kernel: np.ndarray = None) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8) if kernel is None else kernel
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def plot_frames(
    frames: np.ndarray, path: str | Path, index: np.ndarray, prefix: str = None, scale: bool = False
) -> None:
    for frame, i in zip(frames, index):
        amp = max(abs(frame.max()), abs(frame.min())) if scale else None
        filename = index if not prefix else prefix + str(i) + ".png"
        vmin = -amp if amp is not None else None
        plt.imsave(path / filename, frame, cmap="RdBu", vmin=vmin, vmax=amp)


def lineplot_minmax(frames: xr.DataArray, calc_dims: tuple[str, ...], path: Path | str) -> None:
    plt.plot(frames.max(calc_dims))
    plt.plot(frames.min(calc_dims))
    plt.savefig(path)
    plt.close()


def twin_naming(detected_on: np.ndarray) -> Generator[str, None, None]:
    """
    naming scheme that separates cells detected at the same frame with "_{duplicate_idx}" suffix.
    """
    counts = Counter(detected_on)
    seen = defaultdict(int)

    for v in detected_on:
        seen[v] += 1
        base = str(v)

        if counts[v] > 1 and seen[v] > 1:
            yield f"{base}_{seen[v]}"
        else:
            yield base


def footprint_rims(footprints: list[xr.DataArray]) -> np.ndarray:
    boundary = []
    for fp in footprints:
        eroded = erode_to_average(fp.values, "quantile", low=0.1, mask=fp.values > 0, q=0.8)
        border = mark_boundary(eroded, kernel_size=3)
        boundary.append(border > (border.max() / 3))

    all_bounds = np.max(boundary, axis=0)

    return all_bounds
