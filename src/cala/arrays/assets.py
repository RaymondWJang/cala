import contextlib
import shutil
from copy import deepcopy
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


class Asset(BaseModel):
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

    def __eq__(self, other: "Asset") -> bool:
        if isinstance(other, Asset):
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


class Footprint(Asset):
    xr_schema: ClassVar[Schema] = Schema(
        name="footprint",
        dims=(Dims.width.value, Dims.height.value),
        dtype=float,
        checks=[is_non_negative, has_no_nan],
    )


class Trace(Asset):
    xr_schema: ClassVar[Schema] = Schema(
        name="trace",
        dims=(Dims.frame.value,),
        dtype=float,
        checks=[is_non_negative],
    )


class Frame(Asset):
    xr_schema: ClassVar[Schema] = Schema(
        name="frame",
        dims=(Dims.width.value, Dims.height.value),
        dtype=None,  # np.number,  # gets converted to float64 in xarray-validate
        checks=[is_non_negative, has_no_nan],
    )


class Footprints(Asset):
    xr_schema: ClassVar[Schema] = Bundle(
        name="footprint-group",
        member=Footprint.entity(),
        group_by=Dims.component,
        checks=[is_non_negative, has_no_nan],
        allow_extra_coords=False,
    )

    sparsify = True


class Traces(Asset):
    peek_size: int = None
    """How many epochs to return when called."""
    flush_interval: int | None = None
    """How many epochs to wait until next flush"""

    _deprecated: list[str] = PrivateAttr(default_factory=list)
    """
    Deprecated, or replaced component idx.
    Since zarr does not support efficiently removing rows and columns,
    there's no easy way to remove a column when a component has been 
    removed or replaced. Instead, we "mask" it with this "deprecated"
    flag.

    When arrays are called, these are filtered out. When new epochs are
    added, these are added in with nan values. 
    """

    xr_schema: ClassVar[Schema] = Bundle(
        name="trace-group",
        member=Trace.entity(),
        group_by=Dims.component,
        checks=[is_non_negative],
        allow_extra_coords=False,
    )

    @model_validator(mode="after")
    def flush_conditions(self) -> Self:
        assert (self.flush_interval and self.zarr_path) or (
            not self.flush_interval and not self.zarr_path
        ), "zarr_path and flush_interval should either be both provided or neither."
        if self.flush_interval:
            assert self.flush_interval > self.peek_size, (
                f"flush_interval must be larger than peek_size. "
                f"Provided: {self.flush_interval = }, {self.peek_size = }"
            )
        return self

    @property
    def sizes(self) -> dict[str, int]:
        if self.zarr_path:
            total_size = {}
            for key, val in self.array_.sizes.items():
                if key == AXIS.frame_dim:
                    total_size[key] = val + self.load_zarr().sizes[key]
                else:
                    total_size[key] = val
            return total_size
        else:
            return self.array_.sizes

    @property
    def array(self) -> xr.DataArray:
        return (
            self.array_.isel({AXIS.frame_dim: slice(-self.peek_size, None)})
            if self.array_ is not None
            else self.array_
        )

    @array.setter
    def array(self, array: xr.DataArray) -> None:
        """
        In case zarr_path is defined, if array is larger than peek_size,
        the epochs older than -peek_size gets flushed to zarr array.

        """
        if self.validate_schema:
            array.validate.against_schema(self._entity.model)
        if self.zarr_path:
            self.array_ = array.isel({AXIS.frame_dim: slice(-self.peek_size, None)})
            array.isel({AXIS.frame_dim: slice(None, -self.peek_size)}).to_zarr(
                self.zarr_path, mode="w"
            )
        else:
            self.array_ = array

    def append(self, array: xr.DataArray, dim: str) -> None:
        """
        Since we cannot simply append to zarr array in memory using xarray syntax,
        we provide a convenience method for appending to zarr array and in-memory array
        in a streamlined manner.

        Incoming arrays have to be 2-dimensional.

        """

        if dim == AXIS.frame_dim:
            self.array_ = xr.concat([self.array_, array], dim=AXIS.frame_dim)

            if self.zarr_path and self.array_.sizes[AXIS.frame_dim] > self.flush_interval:
                self._flush_zarr()

        elif dim == AXIS.component_dim:
            if self.zarr_path:
                n_in_memory = self.array_.sizes[AXIS.frame_dim]
                self.array_ = xr.concat(
                    [self.array_, array.isel({AXIS.frame_dim: slice(-n_in_memory, None)})],
                    dim=dim,
                )
                array.isel({AXIS.frame_dim: slice(None, -n_in_memory)}).to_zarr(
                    self.zarr_path, append_dim=dim
                )
            else:
                self.array_ = xr.concat([self.array_, array], dim=dim, combine_attrs="drop")

    def _flush_zarr(self) -> None:
        """
        Flushes traces older than peek_size to zarr array.
        Needs to append nans to deprecated components, since they get deleted
        in in-memory array, but persist in zarr array.


        Could do this much more elegantly by pre-allocating .array_
        """
        raw_zarr = self.load_zarr()
        to_flush = self.array_.isel({AXIS.frame_dim: slice(None, -self.peek_size)})
        if self._deprecated:
            zarr_ids = raw_zarr[AXIS.id_coord].values
            zarr_detects = raw_zarr[AXIS.detect_coord].values
            intact_mask = ~np.isin(zarr_ids, self._deprecated)
            n_flush = to_flush.sizes[AXIS.frame_dim]
            prealloc = xr.DataArray(
                np.full((raw_zarr.sizes[AXIS.component_dim], n_flush), np.nan),
                dims=[AXIS.component_dim, AXIS.frame_dim],
                coords={
                    AXIS.id_coord: (AXIS.component_dim, zarr_ids),
                    AXIS.detect_coord: (AXIS.component_dim, zarr_detects),
                },
            ).assign_coords(to_flush[AXIS.frame_dim].coords)
            prealloc.loc[intact_mask] = to_flush
            prealloc.to_zarr(self.zarr_path, append_dim=AXIS.frame_dim)
        else:
            to_flush.to_zarr(self.zarr_path, append_dim=AXIS.frame_dim)
        self.array_ = self.array_.isel({AXIS.frame_dim: slice(-self.peek_size, None)})

    def deprecate_except(self, intact_mask: np.ndarray) -> None:
        if self.zarr_path:
            self._deprecated.extend(self.array_[AXIS.id_coord].values[~intact_mask])
        self.array_ = self.array_[intact_mask]

    @classmethod
    def from_array(cls, array: xr.DataArray) -> "Traces":
        """
        This is only really used for typing / auto-validation purposes,
        so we don't really have to worry about specifying the parameters.

        """
        new_cls = cls(peek_size=array.sizes[AXIS.frame_dim])
        new_cls.array = array
        return new_cls

    def full_array(self, isel_filter: dict = None, sel_filter: dict = None) -> xr.DataArray:
        if self.zarr_path:
            raw_zarr = self.load_zarr(isel_filter, sel_filter)
            zarr_ids = raw_zarr[AXIS.id_coord].values
            intact_mask = ~np.isin(zarr_ids, self._deprecated)

            return xr.concat([raw_zarr[intact_mask], self.array_], dim=AXIS.frame_dim).compute()
        else:
            return self.array_.isel(isel_filter).sel(sel_filter)


class Movie(Asset):
    xr_schema: ClassVar[Schema] = Bundle(
        name="movie",
        member=Frame.entity(),
        group_by=Dims.frame.value,
        checks=[is_non_negative, has_no_nan],
        allow_extra_coords=False,
    )


class PopSnap(Asset):
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


comp_dims = (Dims.component.value, deepcopy(Dims.component.value))
comp_dims[1].name = AXIS.duplicate(comp_dims[1].name)
for coord in comp_dims[1].coords:
    coord.name = AXIS.duplicate(coord.name)


class CompStats(Asset):
    xr_schema: ClassVar[Schema] = Schema(
        name="comp-stat",
        dims=comp_dims,
        dtype=float,
        checks=[is_non_negative, has_no_nan],
        allow_extra_coords=False,
    )


class PixStats(Asset):
    xr_schema: ClassVar[Schema] = Schema(
        name="pix-stat",
        dims=(Dims.width.value, Dims.height.value, Dims.component.value),
        dtype=float,
        checks=[is_non_negative, has_no_nan],
        allow_extra_coords=False,
    )


class Overlaps(Asset):
    xr_schema: ClassVar[Schema] = Schema(
        name="overlap",
        dims=comp_dims,
        dtype=bool,
        checks=[has_no_nan],
        allow_extra_coords=False,
    )


class Buffer(Asset):
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


class Energy(Asset):
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
