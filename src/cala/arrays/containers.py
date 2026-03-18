import contextlib
import shutil
from pathlib import Path
from typing import ClassVar, Self, TypeVar

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator
from sparse import COO

from cala.arrays.axis import AXIS
from cala.arrays.validate import Bundle, Coords, Dims, Schema, has_no_nan, is_non_negative
from cala.config import config
from cala.util import clear_dir

AssetType = TypeVar("AssetType", xr.DataArray, Path, None)


class ArrayContainer(BaseModel):
    array_: AssetType = None
    sparsify: ClassVar[bool] = False
    zarr_path: Path | None = None
    """relative to config.user_data_dir"""
    xr_schema: ClassVar[Schema]
    validate_schema: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @property
    def array(self) -> AssetType:
        return self.array_

    @array.setter
    def array(self, value: xr.DataArray) -> None:
        if self.validate_schema:
            value.validate.against_schema(self.xr_schema.model)
        if self.sparsify and isinstance(value.data, np.ndarray):
            value.data = COO.from_numpy(value.data)
        self.array_ = value

    @classmethod
    def from_array(cls, array: xr.DataArray) -> Self:
        if cls.sparsify and isinstance(array.data, np.ndarray):
            array.data = COO.from_numpy(array.data)
        return cls(array_=array)

    def reset(self) -> None:
        self.array_ = None
        if self.zarr_path:
            path = Path(self.zarr_path)
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                contextlib.suppress(FileNotFoundError)

    def __eq__(self, other: "ArrayContainer") -> bool:
        if isinstance(other, ArrayContainer):
            return self.array.equals(other.array)
        else:
            return False

    @classmethod
    def entity(cls) -> Schema:
        return cls.xr_schema

    @model_validator(mode="after")
    def validate_array_schema(self) -> Self:
        if self.validate_schema and self.array_ is not None:
            self.array_.validate.against_schema(self.xr_schema.model)

        return self

    @field_validator("zarr_path", mode="after")
    @classmethod
    def validate_zarr_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return value
        zarr_dir = (config.user_dir / value).resolve()
        zarr_dir.mkdir(parents=True, exist_ok=True)
        clear_dir(zarr_dir)
        return zarr_dir

    def load_zarr(self, isel_filter: dict = None, sel_filter: dict = None) -> xr.DataArray:
        da = (
            xr.open_zarr(self.zarr_path)
            .isel(isel_filter)
            .sel(sel_filter)
            .to_dataarray()
            .drop_vars(["variable"])
            .isel(variable=0)
        )
        return da.assign_coords(
            {
                AXIS.id_coord: lambda ds: da[AXIS.id_coord].astype(str),
                AXIS.timestamp_coord: lambda ds: da[AXIS.timestamp_coord].astype(str),
            }
        )


class Frame(ArrayContainer):
    xr_schema: ClassVar[Schema] = Schema(
        name="frame",
        dims=(Dims.width.value, Dims.height.value),
        dtype=None,  # np.number,  # gets converted to float64 in xarray-validate
        checks=[is_non_negative, has_no_nan],
    )


class Movie(ArrayContainer):
    xr_schema: ClassVar[Schema] = Bundle(
        name="movie",
        member=Frame.entity(),
        group_by=Dims.frame.value,
        checks=[is_non_negative, has_no_nan],
        allow_extra_coords=False,
    )


class Buffer(ArrayContainer):
    """
    Implements a bip buffer to avoid expensive copying that occurs with
    numpy concat, append, and stack.

    Works by preallocating a space twice the desired size.
    """

    xr_schema: ClassVar[Schema] = Bundle(
        name="frame",
        member=Frame.entity(),
        group_by=Dims.frame.value,
        checks=[is_non_negative, has_no_nan],
        allow_extra_coords=False,
    )

    validate_schema: bool = False
    """Validation currently does not play nicely with this class."""

    size: int
    _full: bool = PrivateAttr(False)
    _next: int = PrivateAttr(default=0)

    def append(self, array: xr.DataArray) -> None:
        self.array_.data[self._next] = array.data
        self.array_.data[self._next + self.size] = array.data
        for coord in [AXIS.frame_coord, AXIS.timestamp_coord]:
            self.array_[coord].data[self._next] = array[coord].item()
            self.array_[coord].data[self._next + self.size] = array[coord].item()

        self._next = (self._next + 1) % self.size
        if not self._full:
            # check if this made the buffer full
            self._full = self._next == 0

    @property
    def array(self) -> xr.DataArray | None:
        if self.array_ is None:
            return None
        if self._full:
            out = self.array_.isel({AXIS.frame_dim: slice(self._next, self._next + self.size)})
        else:
            out = self.array_.isel({AXIS.frame_dim: slice(None, self._next)})
        # kinda expensive. maybe float is fine?
        return out  # .assign_coords({AXIS.frame_coord: out[AXIS.frame_coord].astype(int)})

    @array.setter
    def array(self, array: xr.DataArray) -> None:
        """
        Build a new buffer array.
        """
        array = (
            array.volumize.dim_with_coords(
                dim=AXIS.frame_dim, coords=[AXIS.frame_coord, AXIS.timestamp_coord]
            )
            if AXIS.frame_dim not in array.dims
            else array.isel({AXIS.frame_dim: slice(-self.size, None)})
        )
        fill_sizes = dict(array.sizes)
        fill_sizes[AXIS.frame_dim] = self.size - array.sizes[AXIS.frame_dim]
        fill = np.zeros(list(fill_sizes.values()))
        filler = xr.DataArray(
            fill,
            dims=array.dims,
            coords={
                AXIS.frame_coord: (AXIS.frame_dim, [np.nan] * (fill_sizes[AXIS.frame_dim])),
                AXIS.timestamp_coord: (AXIS.frame_dim, [""] * (fill_sizes[AXIS.frame_dim])),
            },
        )
        buffer = xr.concat([array, filler] * 2, dim=AXIS.frame_dim)

        self._full = array.sizes[AXIS.frame_dim] >= self.size
        self._next = np.min((array.sizes[AXIS.frame_dim], self.size)) % self.size
        self.array_ = buffer

    @classmethod
    def from_array(cls, array: xr.DataArray, size: int) -> Self:
        buffer = cls(size=size)
        buffer.array = array
        return buffer


class Energy(ArrayContainer):
    xr_schema: ClassVar[Schema] = Schema(
        name="energy",
        dims=(Dims.width.value, Dims.height.value),
        dtype=None,  # np.number,  # gets converted to float64 in xarray-validate
        checks=[is_non_negative, has_no_nan],
    )

    _mean: np.ndarray = PrivateAttr(None)
    _sq_mean: np.ndarray = PrivateAttr(None)

    def update_std(self, arr: xr.DataArray) -> xr.DataArray:
        """median: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=614292"""
        eta = 1 / (arr[AXIS.frame_coord].item() + 1)

        if self._mean is None:
            self._mean = arr.values
            self._sq_mean = np.square(arr.values)
        else:
            self._mean += eta * (arr.values - self._mean)
            self._sq_mean += eta * (np.square(arr.values) - self._sq_mean)
        return xr.DataArray(np.sqrt(self._sq_mean - np.square(self._mean)), dims=arr.dims)


class PopSnap(ArrayContainer):
    """
    A snapshot of a population trait.

    Mainly used for Traces that only has one frame.
    """

    xr_schema: ClassVar[Schema] = Schema(
        name="pop-snap",
        dims=(Dims.component.value,),
        dtype=float,
        coords=[Coords.frame.value, Coords.timestamp.value],
        checks=[is_non_negative, has_no_nan],
    )


class Footprint(ArrayContainer):
    xr_schema: ClassVar[Schema] = Schema(
        name="footprint",
        dims=(Dims.width.value, Dims.height.value),
        dtype=float,
        checks=[is_non_negative, has_no_nan],
    )


class Trace(ArrayContainer):
    xr_schema: ClassVar[Schema] = Schema(
        name="trace",
        dims=(Dims.frame.value,),
        dtype=float,
        checks=[is_non_negative],
    )
