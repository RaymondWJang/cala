from abc import ABC, abstractmethod
from copy import deepcopy
from typing import ClassVar, Self, Mapping, Hashable, Any, TypedDict

import numpy as np
import xarray as xr
from pydantic import PrivateAttr, model_validator
from scipy.sparse import vstack, csc_matrix
from sparse import COO
from xarray import Coordinates

from cala.arrays import AXIS
from cala.arrays.containers import ArrayContainer, Footprint, Trace
from cala.arrays.validate import Dims, Schema, is_non_negative, has_no_nan, Bundle
from cala.util import concatenate_coordinates


class ModelMixin(ABC):
    @abstractmethod
    def commit_extension(self, new, inplace: bool = False) -> None | xr.DataArray:
        """
        Commit new components into the model.
        Currently, the non-inplace option returns updated xr.DataArray,
        instead of returning the model wrapping the xr.DataArray.
        This behavior will be fixed in the future
        when I figure out how to copy over the attributes properly.
        """

    @abstractmethod
    def apply_fit(self, new, inplace: bool = False) -> None | xr.DataArray:
        """
        Apply the new calculated fit into the model.
        Currently, the non-inplace option returns updated xr.DataArray,
        instead of returning the model wrapping the xr.DataArray.
        This behavior will be fixed in the future
        when I figure out how to copy over the attributes properly.
        """

    @abstractmethod
    def deprecate(
        self, mask: np.ndarray[Any, np.dtype[np.bool]], inplace: bool = False
    ) -> None | xr.DataArray:
        """
        Deprecate components based on a *mask*.
        Currently, the non-inplace option returns updated xr.DataArray,
        instead of returning the model wrapping the xr.DataArray.
        This behavior will be fixed in the future
        when I figure out how to copy over the attributes properly.
        """


class Footprints(ArrayContainer, ModelMixin):
    xr_schema: ClassVar[Schema] = Bundle(
        name="footprint-group",
        member=Footprint.entity(),
        group_by=Dims.component,
        checks=[is_non_negative, has_no_nan],
        allow_extra_coords=False,
    )

    sparsify = True

    def commit_extension(self, new: "Footprints", inplace: bool = False) -> None | xr.DataArray:
        data = vstack(
            [self.array.data.reshape((self.array.sizes[AXIS.component_dim], -1)).tocsr(), new]
        )
        coords = concatenate_coordinates(
            self.array[AXIS.component_dim].coords, new.array[AXIS.component_dim].coords
        )
        extended = xr.DataArray(
            COO.from_scipy_sparse(data).reshape((data.shape[0], *self.array.shape[1:])),
            dims=self.array.dims,
            coords={k: (AXIS.component_dim, v) for k, v in coords.items()},
        )
        extended = extended.assign_coords(self.array[AXIS.width_coord].coords)
        extended = extended.assign_coords(self.array[AXIS.height_coord].coords)

        if inplace:
            self.array = extended
            return None
        else:
            return extended

    def apply_fit(self, new: csc_matrix, inplace: bool = False) -> None | xr.DataArray:
        """
        Not much to do to apply_fit, since the fit is directly applied
        during the fit_one optimization.
        """
        fitted = xr.DataArray(
            COO.from_scipy_sparse(new.T).reshape(self.array.shape),
            dims=self.array.dims,
            coords=self.array.coords,
        )
        if inplace:
            self.array = fitted
            return None
        else:
            return fitted

    def deprecate(
        self, mask: np.ndarray[Any, np.dtype[np.bool]], inplace: bool = False
    ) -> None | xr.DataArray:
        keep_mask = ~mask
        val = self.array.data.reshape((self.array.sizes[AXIS.component_dim], -1)).tocsr()[keep_mask]
        coords = self.array[AXIS.component_dim][keep_mask].coords
        reduced = xr.DataArray(
            COO.from_scipy_sparse(val).reshape((val.shape[0], *self.array.shape[1:])),
            dims=self.array.dims,
            coords=coords,
        )
        reduced = reduced.assign_coords(self.array[AXIS.width_coord].coords)
        reduced = reduced.assign_coords(self.array[AXIS.height_coord].coords)

        if inplace:
            self.array = reduced
            return None
        else:
            return reduced


class Traces(ArrayContainer, ModelMixin):
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
    def sizes(self) -> Mapping[Hashable, int]:
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

    def commit_extension(self, new: xr.DataArray, inplace: bool = False) -> "None | Traces":
        """
        Because of the Zarr implementation,
        new trace object would either need to:
        1. copy the entire Zarr array file, (deepcopy-style) or
        2. simply copy the Zarr array filepath (shallow copy).

        Option 1 is likely too heavy of an implementation and
        I do not see a lot of use cases for it.
        Option 2 implements a very inefficient method,
        where we copy over the entire self to a new Traces instance,
        and update that new instance's _deprecated and array_ attrs.

        TODO: We're unnecessarily copying array_ twice. Must be made more efficient.
        """
        extended = self if inplace else deepcopy(self)

        if extended.zarr_path:
            n_in_memory = extended.array_.sizes[AXIS.frame_dim]
            extended.array_ = xr.concat(
                [extended.array_, new.isel({AXIS.frame_dim: slice(-n_in_memory, None)})],
                dim=AXIS.component_dim,
            )
            new.isel({AXIS.frame_dim: slice(None, -n_in_memory)}).to_zarr(
                extended.zarr_path, append_dim=AXIS.component_dim
            )
        else:
            extended.array_ = xr.concat(
                [extended.array_, new], dim=AXIS.component_dim, combine_attrs="drop"
            )

        return None if inplace else extended

    def apply_fit(self, new: xr.DataArray, inplace: bool = False) -> "None | Traces":
        """
        Because of the Zarr implementation,
        new trace object would either need to:
        1. copy the entire Zarr array file, (deepcopy-style) or
        2. simply copy the Zarr array filepath (shallow copy).

        Option 1 is likely too heavy of an implementation and
        I do not see a lot of use cases for it.
        Option 2 implements a very inefficient method,
        where we copy over the entire self to a new Traces instance,
        and update that new instance's _deprecated and array_ attrs.

        TODO: We're unnecessarily copying array_ twice. Must be made more efficient.
        """
        fitted = self if inplace else deepcopy(self)

        fitted.array_ = xr.concat([fitted.array_, new], dim=AXIS.frame_dim)

        if fitted.zarr_path and fitted.array_.sizes[AXIS.frame_dim] > fitted.flush_interval:
            fitted._flush_zarr()

        return None if inplace else fitted

    def append(self, array: xr.DataArray, dim: str) -> None:
        """
        ! DEPRECATION WARNING: Slated to be replaced by
        ! :meth:`.Traces.commit_extension` and :math:`.Traces.apply_fit`.
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

    def deprecate(
        self, mask: np.ndarray[Any, np.dtype[np.bool]], inplace: bool = False
    ) -> "None | Traces":
        """
        Because of the Zarr implementation,
        new trace object would either need to:
        1. copy the entire Zarr array file, (deepcopy-style) or
        2. simply copy the Zarr array filepath (shallow copy).

        Option 1 is likely too heavy of an implementation and
        I do not see a lot of use cases for it.
        Option 2 implements a very inefficient method,
        where we copy over the entire self to a new Traces instance,
        and update that new instance's _deprecated and array_ attrs.

        TODO: We're unnecessarily copying array_ twice. Must be made more efficient.
        """
        reduced = self if inplace else deepcopy(self)

        if reduced.zarr_path:
            reduced._deprecated.extend(reduced.array_[AXIS.id_coord].values[mask])
        reduced.array_ = reduced.array_[~mask]

        return None if inplace else reduced

    def full_array(self, isel_filter: dict = None, sel_filter: dict = None) -> xr.DataArray:
        if self.zarr_path:
            raw_zarr = self.load_zarr(isel_filter, sel_filter)
            zarr_ids = raw_zarr[AXIS.id_coord].values
            intact_mask = ~np.isin(zarr_ids, self._deprecated)

            return xr.concat([raw_zarr[intact_mask], self.array_], dim=AXIS.frame_dim).compute()
        else:
            return self.array_.isel(isel_filter).sel(sel_filter)

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

    @classmethod
    def from_array(cls, array: xr.DataArray) -> "Traces":
        """
        This is only really used for typing / auto-validation purposes,
        so we don't really have to worry about specifying the parameters.
        TODO: not sure what i meant by above... But this function seems replaceable.
        """
        new_cls = cls(peek_size=array.sizes[AXIS.frame_dim])
        new_cls.array = array
        return new_cls


class PixStats(ArrayContainer, ModelMixin):
    xr_schema: ClassVar[Schema] = Schema(
        name="pix-stat",
        dims=(Dims.width.value, Dims.height.value, Dims.component.value),
        dtype=float,
        checks=[is_non_negative, has_no_nan],
        allow_extra_coords=False,
    )

    def commit_extension(self, new: xr.DataArray, inplace: bool = False) -> None | xr.DataArray:
        extended = xr.concat([self.array, new], dim=AXIS.component_dim)
        if inplace:
            self.array = extended
            return None
        else:
            return extended

    def apply_fit(self, new: xr.DataArray, inplace: bool = False) -> None | xr.DataArray:
        """
        Not much to do to apply_fit, since the fit is directly applied
        during the fit_one optimization.
        """
        CY = new.reshape(self.array.shape)
        fitted = xr.DataArray(CY, dims=self.array.dims, coords=self.array.coords)
        if inplace:
            self.array = fitted
            return None
        else:
            return fitted

    def deprecate(
        self, mask: np.ndarray[Any, np.dtype[np.bool]], inplace: bool = False
    ) -> None | xr.DataArray:
        reduced = self.array[~mask]
        if inplace:
            self.array = reduced
            return None
        else:
            return reduced


comp_dims = (Dims.component.value, deepcopy(Dims.component.value))
comp_dims[1].name = AXIS.duplicate(comp_dims[1].name)
for coord in comp_dims[1].coords:
    coord.name = AXIS.duplicate(coord.name)


class MatrixExtension(TypedDict):
    tr_block: xr.DataArray | tuple[np.ndarray, np.ndarray]
    bl_block: xr.DataArray | tuple[np.ndarray, np.ndarray]
    tps_block: xr.DataArray | tuple[np.ndarray, np.ndarray]
    ext_coords: Coordinates


class CompStats(ArrayContainer, ModelMixin):
    xr_schema: ClassVar[Schema] = Schema(
        name="comp-stat",
        dims=comp_dims,
        dtype=float,
        checks=[is_non_negative, has_no_nan],
        allow_extra_coords=False,
    )

    def apply_fit(self, new: xr.DataArray, inplace: bool = False) -> None | xr.DataArray:
        epoch = new[AXIS.frame_coord].item()
        scale = epoch / (epoch + 1)
        fitted = (scale * self.array + (1 - scale) * new).reset_coords(
            [AXIS.timestamp_coord, AXIS.frame_coord], drop=True
        )
        if inplace:
            self.array = fitted
            return None
        else:
            return fitted

    def commit_extension(self, new: MatrixExtension, inplace: bool = False) -> None | xr.DataArray:
        # Top block: [leading principal submatrix, top right block]
        top_block = xr.concat([self.array, new["tr_block"]], dim=AXIS.component_dim)
        # Bottom block: [bottom left block, trailing principal submatrix]
        bottom_block = xr.concat([new["bl_block"], new["tps_block"]], dim=AXIS.component_dim)
        # Combine blocks
        extended = xr.concat([top_block, bottom_block], dim=AXIS.duplicate(AXIS.component_dim))
        if inplace:
            self.array = extended
            return None
        else:
            return extended

    def deprecate(
        self, mask: np.ndarray[Any, np.dtype[np.bool]], inplace: bool = False
    ) -> None | xr.DataArray:
        keep_mask = ~mask
        reduced = self.array[keep_mask].T[keep_mask]
        if inplace:
            self.array = reduced
            return None
        else:
            return reduced


def overlap_format(array: COO, V_comp: xr.DataArray, a_new_comp: Coordinates) -> xr.DataArray:
    prim_coords = concatenate_coordinates(V_comp.coords, a_new_comp)
    seco_coords = concatenate_coordinates(
        V_comp.rename(AXIS.component_rename).coords,
        a_new_comp.to_dataset().rename(AXIS.component_rename).coords,
    )
    return xr.DataArray(
        array,
        dims=(AXIS.component_dim, AXIS.duplicate(AXIS.component_dim)),
        coords={k: (AXIS.component_dim, v) for k, v in prim_coords.items()},
    ).assign_coords({k: (AXIS.duplicate(AXIS.component_dim), v) for k, v in seco_coords.items()})


def assemble_sparse_bool(
    top_left: tuple[np.ndarray, np.ndarray],
    top_right: tuple[np.ndarray, np.ndarray],
    bottom_left: tuple[np.ndarray, np.ndarray],
    bottom_right: tuple[np.ndarray, np.ndarray],
    init_shape: tuple[int, int],
    attach_shape: tuple[int, int],
) -> COO:
    """
    Assemble a sparse boolean array with four coordinates in the format of
    scipy.sparse.sp_matrix.nonzero()

    """
    x_coords = top_left[0]
    y_coords = top_left[1]
    x_coords = np.concatenate([x_coords, top_right[0]])
    y_coords = np.concatenate([y_coords, top_right[1] + init_shape[1]])
    x_coords = np.concatenate([x_coords, bottom_left[0] + init_shape[0]])
    y_coords = np.concatenate([y_coords, bottom_left[1]])
    x_coords = np.concatenate([x_coords, bottom_right[0] + init_shape[0]])
    y_coords = np.concatenate([y_coords, bottom_right[1] + init_shape[1]])

    final_shape = tuple(x1 + x2 for x1, x2 in zip(init_shape, attach_shape))
    return COO(coords=(x_coords, y_coords), shape=final_shape, data=1)


class Overlaps(ArrayContainer, ModelMixin):
    xr_schema: ClassVar[Schema] = Schema(
        name="overlap",
        dims=comp_dims,
        dtype=bool,
        checks=[has_no_nan],
        allow_extra_coords=False,
    )

    def apply_fit(self, new: xr.DataArray, inplace: bool = False) -> None | xr.DataArray:
        """
        Overlaps update from new footprints fits is not implemented
        """
        raise NotImplementedError("Overlaps update from new footprint fits is not implemented")

    def commit_extension(self, new: MatrixExtension, inplace: bool = False) -> None | xr.DataArray:
        extended = assemble_sparse_bool(
            self.array.data.tocsr().nonzero(),
            new["tr_block"].nonzero(),
            new["bl_block"].nonzero(),
            new["tps_block"].nonzero(),
            self.array.data.tocsr().shape,
            new["tps_block"].shape,
        )
        extended = overlap_format(extended, self.array[AXIS.component_dim], new["ext_coords"])
        if inplace:
            self.array = extended
            return None
        else:
            return extended

    def deprecate(
        self, mask: np.ndarray[Any, np.dtype[np.bool]], inplace: bool = False
    ) -> None | xr.DataArray:
        keep_mask = ~mask
        reduced = self.array.data.tocsr()[keep_mask].T[keep_mask]
        reduced = xr.DataArray(
            COO.from_scipy_sparse(reduced),
            dims=self.array.dims,
            coords=self.array[AXIS.component_dim][keep_mask].coords,
        )
        reduced = reduced.assign_coords(
            self.array[AXIS.component_dim][keep_mask].rename(AXIS.component_rename).coords
        )
        if inplace:
            self.array = reduced
            return None
        else:
            return reduced
