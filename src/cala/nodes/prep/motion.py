import functools
from collections.abc import Callable
from typing import Annotated as A

import cv2
import numpy as np
import xarray as xr
from noob import Name, process_method
from numpydantic import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from skimage.filters import difference_of_gaussians

from cala.arrays import AXIS, Frame


class Shift(BaseModel):
    height: float
    width: float

    @classmethod
    def from_arr(cls, array: NDArray) -> "Shift":
        assert array.shape == (2,)
        return Shift(height=array[0], width=array[1])

    def __add__(self, other: "Shift") -> "Shift":
        return Shift(height=self.height + other.height, width=self.width + other.width)


class Anchor(BaseModel):
    max_shift_w: int = 20
    max_shift_h: int = 20

    dog_kwargs: dict = Field(default_factory=dict, validate_default=True)
    gauss_kwargs: dict = Field(default_factory=dict, validate_default=True)

    _reg_shift: Callable = PrivateAttr(None)
    """A callable used to find the shift"""
    _local: xr.DataArray = PrivateAttr(None)
    """local anchor - processed and ready for comparison"""
    _global: xr.DataArray = PrivateAttr(None)
    """global anchor - processed and ready for comparison"""
    _history: list[Shift] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("dog_kwargs", mode="before")
    @classmethod
    def default_dog(cls, value: dict) -> dict:
        if not value:
            return {"low_sigma": 3}
        else:
            return value

    @field_validator("gauss_kwargs", mode="before")
    @classmethod
    def default_gauss(cls, value: dict) -> dict:
        if not value:
            return {"ksize": (11, 11), "sigmaX": 20}
        else:
            return value

    @process_method
    def stabilize(self, frame: Frame) -> A[Frame, Name("frame")]:
        """
        --- image, prepped, local ---
        image: original image. only shifted and outputted
        prepped: processed image. only used to find the shift and then discarded
        local: shifted prepped from the last iteration to be used as a template

        Steps:
        1. raw prepped gets shifted to last local anchor. We save the shift
        2. the shifted prepped gets shifted to global anchor. We add to the shift
        3. we apply total shift to image. We also save the total shifted prepped
           as the anchor for the next frame
        """
        arr = frame.array
        prepped = _prepare(arr, dog_kwargs=self.dog_kwargs, gauss_kwargs=self.gauss_kwargs)
        if not self._has_prereqs:
            self._init(prepped)
            return frame

        total = Shift(height=0, width=0)
        for template in [self._local, self._global]:
            shift_arr = match_template(prepped.values, template.values)
            # shift_arr, _, _ = self._reg_shift(template.values, prepped.values)
            shift = Shift.from_arr(shift_arr)
            total += shift
            prepped = apply_shift(prepped, shift)
        self._get_ready_for_next(prepped)
        self._history.append(total)

        result = apply_shift(arr, total)
        return Frame.from_array(result)

    @property
    def _has_prereqs(self) -> bool:
        return self._local is not None

    def _init(self, image: xr.DataArray) -> None:
        self._local = image
        self._global = image
        self._reg_shift = functools.partial(
            match_template, max_shift_w=self.max_shift_w, max_shift_h=self.max_shift_h
        )

    def _calculate_shift(self, shift: xr.DataArray) -> xr.DataArray: ...

    def _get_ready_for_next(self, prepped: xr.DataArray) -> None:
        self._local = prepped

        # global learns the local
        curr_idx = prepped[AXIS.frame_coord].item()
        self._global = (self._global * curr_idx + self._local) / (curr_idx + 1)


def _prepare(image: xr.DataArray, dog_kwargs: dict, gauss_kwargs: dict) -> xr.DataArray:
    tmp = difference_of_gaussians(image, **dog_kwargs)
    tmp = cv2.normalize(tmp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result = cv2.GaussianBlur(tmp.astype(float), **gauss_kwargs)
    return xr.DataArray(result, dims=image.dims, coords=image.coords)


def match_template(
    image: np.ndarray, template: np.ndarray, max_shift_w: int = 10, max_shift_h: int = 10
) -> np.ndarray:
    """https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html"""
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    height, width = template.shape
    max_shift_rows = max_shift_h
    max_shift_cols = max_shift_w

    templ_crop = template[
        max_shift_rows : height - max_shift_rows, max_shift_cols : width - max_shift_cols
    ].astype(np.float32)

    res = cv2.matchTemplate(image, templ_crop, cv2.TM_CCORR_NORMED)
    peak_col, peak_row = cv2.minMaxLoc(res)[3]

    if (0 < peak_row < 2 * max_shift_rows - 1) & (0 < peak_col < 2 * max_shift_cols - 1):
        # if max is internal, check for subpixel shift using gaussian peak registration

        # Near its maximum, a correlation peak often looks approximately like a 2D Gaussian bump
        # $R(\Delta)\approx A\exp(-\frac{(\Delta-\Delta_0)^2}{2\sigma^2})$
        # Taking a log turns it into quadratic:
        # $\log{R(\Delta)}\approx \log{A}-\frac{(\Delta-\Delta_0)^2}{2\sigma^2}$
        # So if you look at $\log{R}$ at the peak and its immediate neighbors,
        # you can fit a 1D quadratic in each axis and
        # estimate the subpixel offset of the true maximum.

        # For a 1D function sampled at -1, 0, +1:
        # Let $a = \log{R(-1)}, b = \log{R(0)}, c = \log{R(+1)}$
        # Then the subpixel offset relative to 0 for the parabola's maximum is:
        # $\delta = \frac{a-c}{2(a-2b+c)}
        log_up = np.log(res[peak_row - 1, peak_col])
        log_down = np.log(res[peak_row + 1, peak_col])
        log_left = np.log(res[peak_row, peak_col - 1])
        log_right = np.log(res[peak_row, peak_col + 1])
        log_center = 4 * np.log(res[peak_row, peak_col])

        row_shift = -(
            peak_row  # integer shift
            - max_shift_rows  # because of crop
            + ((log_up - log_down) / (2 * log_up - log_center + 2 * log_down))  # subpixel shift
        )
        col_shift = -(
            peak_col
            - max_shift_cols
            + ((log_left - log_right) / (2 * log_left - log_center + 2 * log_right))
        )
    else:
        row_shift = -(peak_row - max_shift_rows)
        col_shift = -(peak_col - max_shift_cols)

    return np.array([row_shift, col_shift])


def apply_shift(image: xr.DataArray, shift: Shift) -> xr.DataArray:
    M = np.float32([[1, 0, shift.width], [0, 1, shift.height]])

    shifted_frame = cv2.warpAffine(
        image.values,
        M,
        (image.sizes[AXIS.width_dim], image.sizes[AXIS.height_dim]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return xr.DataArray(shifted_frame, dims=image.dims, coords=image.coords)
