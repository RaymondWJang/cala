from collections.abc import Iterable
from itertools import compress
from typing import Annotated as A

import cv2
import numpy as np
import xarray as xr
from noob import Name
from noob.node import Node
from pydantic import Field
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import label
from xarray import Coordinates

from cala.arrays import AXIS, Footprint, Footprints, Trace, Traces
from cala.util import combine_attr_replaces, concat_components, create_id, rank1nmf


class Cataloger(Node):
    trace_smooth_kwargs: dict
    shape_smooth_kwargs: dict
    age_limit: int
    """Don't merge with new components if older than this number of frames."""
    merge_threshold: float
    val_threshold: float = Field(gt=0, lt=1)
    cnt_threshold: int = Field(gt=0)
    """must have cnt-number of pixels that are above the val-value"""

    def process(
        self,
        new_fps: list[Footprint],
        new_trs: list[Trace],
        existing_fp: Footprints | None = None,
        existing_tr: Traces | None = None,
    ) -> tuple[A[Footprints, Name("new_footprints")], A[Traces, Name("new_traces")]]:

        no_new_components = not new_fps or not new_trs
        if no_new_components:
            return Footprints(), Traces()

        shape_chunks = xr.concat([fp.array for fp in new_fps], dim=AXIS.component_dim)
        trace_chunks = xr.concat([tr.array for tr in new_trs], dim=AXIS.component_dim)
        merge_mat = self._monopartite_merge_matrix(shape_chunks, trace_chunks)
        new_shapes, new_traces = self._merge_candidates(shape_chunks, trace_chunks, merge_mat)

        known_fp, known_tr = _get_absorption_targets(
            existing_fp.array, existing_tr.array, self.age_limit
        )
        is_absorbing = known_fp is not None and known_fp.size > 0
        merge_groups = None
        merged_fps = []
        merged_trs = []

        if is_absorbing:
            merge_groups = self._bipartite_merge_groups(new_shapes, new_traces, known_fp, known_tr)
            merged_fps, merged_trs = self._absorb(
                new_shapes, new_traces, known_fp, known_tr, merge_groups
            )
        discrete_fps, discrete_trs = _gather_discrete(new_shapes, new_traces, merge_groups)
        filtered_fps, filtered_trs = self._quality_control(
            [*discrete_fps, *merged_fps], [*discrete_trs, *merged_trs]
        )
        none_passed = not filtered_fps
        if none_passed:
            return Footprints(), Traces()

        footprints, traces = concat_components(
            filtered_fps, filtered_trs, [AXIS.id_coord, AXIS.detect_coord], combine_attr_replaces
        )

        return Footprints.from_array(footprints), Traces.from_array(traces)

    def _merge_matrix(
        self,
        shapes_1: xr.DataArray,
        traces_1: xr.DataArray,
        shapes_2: csr_matrix | xr.DataArray,
        traces_2: xr.DataArray,
    ) -> xr.DataArray:
        """
        Calculates whether two sets of components are mergeable or not

        """
        overlaps = shapes_1.data @ shapes_2.T > 0
        # corr is fast. (~1ms to 4ms)
        corrs = xr.corr(traces_1, traces_2, dim=AXIS.frame_dim) > self.merge_threshold
        return xr.DataArray(overlaps * corrs.values, dims=corrs.dims, coords=corrs.coords)

    def _monopartite_merge_matrix(self, fps: xr.DataArray, trs: xr.DataArray) -> xr.DataArray:
        """
        Calculate the merge matrix for a single set of components.
        Returns a boolean matrix, where the coordinates (i, j) of True values
        mean the two components of index (i, j) are mergeable.

        """
        fps = fps.stack(pixels=AXIS.spatial_dims)
        fps2 = fps.data

        smooth_trs = _smooth_traces(trs, self.trace_smooth_kwargs)
        trs2 = smooth_trs.rename({AXIS.component_dim: AXIS.duplicate(AXIS.component_dim)})
        return self._merge_matrix(fps, smooth_trs, fps2, trs2)

    def _merge_candidates(
        self, footprints: xr.DataArray, traces: xr.DataArray, merge_matrix: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Merge a single set of candidate components with each other,
        according to the merge_matrix.

        """
        num, labels = connected_components(merge_matrix.data)
        combined_fps = []
        combined_trs = []

        for lbl in set(labels):
            group = np.where(labels == lbl)[0]
            fps = footprints.sel({AXIS.component_dim: group})
            trs = traces.sel({AXIS.component_dim: group})
            if len(group) > 1:
                mov = xr.DataArray(
                    _create_component_movie(fps, trs), dims=[*AXIS.spatial_dims, AXIS.frame_dim]
                )
                new_fp, new_tr = self._recompose(mov, footprints[0].coords, traces[0].coords)
            else:
                new_fp, new_tr = fps[0], trs[0]
            combined_fps.append(new_fp)
            combined_trs.append(new_tr)

        new_fps, new_trs = concat_components(combined_fps, combined_trs)

        return new_fps, new_trs

    def _bipartite_merge_groups(
        self,
        candidate_fps: xr.DataArray,
        candidate_trs: xr.DataArray,
        target_fps: csr_matrix,
        target_trs: xr.DataArray,
    ) -> xr.DataArray:
        """
        Calculate the merge groups for two sets of components.
        Returns an integer matrix, where the components (i, j) that correspond to the
        coordinates of a nonzero integer values (n) belong to the group (n).

        """
        fps1 = candidate_fps.stack(pixels=AXIS.spatial_dims)
        trs1 = _smooth_traces(candidate_trs, self.trace_smooth_kwargs)
        smooth_ante = _smooth_traces(target_trs, self.trace_smooth_kwargs)
        trs2 = smooth_ante.rename(AXIS.component_rename)

        mat = self._merge_matrix(fps1, trs1, target_fps, trs2)

        mat.data = label(mat.to_numpy(), background=0, connectivity=1)
        mat = mat.assign_coords(
            {AXIS.component_dim: range(mat.sizes[AXIS.component_dim])}
        ).reset_index(AXIS.component_dim)

        return mat

    def _absorb(
        self,
        new_fps: xr.DataArray,
        new_trs: xr.DataArray,
        known_fps: csr_matrix,
        known_trs: xr.DataArray,
        merge_matrix: xr.DataArray,
    ) -> tuple[list[xr.DataArray], list[xr.DataArray]]:
        footprints = []
        traces = []

        num_merge_groups = merge_matrix.max().item()
        if num_merge_groups > 0:
            for lbl in range(1, num_merge_groups + 1):
                new_idx, known_idx = np.where(merge_matrix == lbl)
                new_idx, known_idx = np.unique(new_idx), np.unique(known_idx)
                fp = new_fps.sel({AXIS.component_dim: new_idx})
                tr = new_trs.sel({AXIS.component_dim: new_idx})
                footprint, trace = self._absorb_component(fp, tr, known_fps, known_trs, known_idx)

                footprints.append(footprint)
                traces.append(trace)

        return footprints, traces

    def _absorb_component(
        self,
        insert_fps: xr.DataArray,
        insert_trs: xr.DataArray,
        absorber_fps: csr_matrix,
        absorber_trs: xr.DataArray,
        absorber_idx: Iterable[int],
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Absorb new components into target component by performing
        rank-1 NMF against a combined movie of the two components

        Adds attribute "replaces" to indicate the target component
        involved in the process, which later will to be replaced by the
        new combined component.

        """
        absorber_fp = absorber_fps[absorber_idx]
        absorber_tr = absorber_trs[absorber_idx]

        absorber_movie = _create_component_movie(absorber_fp, absorber_tr)
        insert_movie = _create_component_movie(insert_fps, insert_trs)

        combined_movie = xr.DataArray(
            absorber_movie.reshape(insert_movie.shape) + insert_movie,
            dims=[*AXIS.spatial_dims, AXIS.frame_dim],
        )

        a_new, c_new = self._recompose(
            combined_movie,
            insert_fps.isel({AXIS.component_dim: 0}).coords,
            insert_trs.isel({AXIS.component_dim: 0}).coords,
        )
        a_new.attrs["replaces"] = absorber_tr[AXIS.id_coord].values.tolist()
        c_new.attrs["replaces"] = absorber_tr[AXIS.id_coord].values.tolist()

        return _register(a_new, c_new)

    def _quality_control(
        self, footprints: xr.DataArray, traces: xr.DataArray
    ) -> tuple[list[xr.DataArray], list[xr.DataArray]]:
        """
        Filters resulting footprints and traces based on quality thresholds

        """
        mask = [
            np.sum(fp.data / fp.data.max() > self.val_threshold) > self.cnt_threshold
            for fp in footprints
        ]
        footprints = list(compress(footprints, mask))
        traces = list(compress(traces, mask))

        return footprints, traces

    def _recompose(
        self, movie: xr.DataArray, fp_coords: Coordinates, tr_coords: Coordinates
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Recompose the movie to a single component using a rank-1 NMF,
        with the coordinates of the absorbing component.

        """
        movie = movie.assign_coords({ax: movie[ax] for ax in AXIS.spatial_dims})
        shape = xr.DataArray(
            np.sum(movie.transpose(AXIS.frame_dim, ...).data, axis=0) > 0, dims=AXIS.spatial_dims
        )
        slice_ = movie.where(shape.as_numpy(), 0, drop=True)
        R = slice_.stack(space=AXIS.spatial_dims).transpose("space", AXIS.frame_dim)

        a, c, error = rank1nmf(R.values, np.mean(R.values, axis=1))

        a_new, c_new = self._reshape(
            footprint=a,
            trace=c,
            fp_coords=fp_coords,
            tr_coords=tr_coords,
            slice_coords=slice_.coords,
        )

        factor = slice_.data.max() / c_new.data.max()
        a_new = a_new / factor
        c_new = c_new * factor

        return a_new, c_new

    def _reshape(
        self,
        footprint: np.ndarray,
        trace: np.ndarray,
        fp_coords: Coordinates,
        tr_coords: Coordinates,
        slice_coords: Coordinates,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Convert back to xarray with proper dimensions and coordinates"""

        c_new = xr.DataArray(trace.squeeze(), dims=[AXIS.frame_dim], coords=tr_coords)

        a_new = xr.DataArray(
            np.zeros(tuple(fp_coords.sizes.values())),
            dims=tuple(fp_coords.sizes.keys()),
            coords=fp_coords,
        )

        smooth_footprint = _smooth_footprint(
            footprint.squeeze().reshape(list(slice_coords[ax].size for ax in AXIS.spatial_dims)),
            self.shape_smooth_kwargs,
        )

        a_new.loc[slice_coords] = xr.DataArray(
            smooth_footprint, dims=AXIS.spatial_dims, coords=slice_coords
        )

        return a_new, c_new


def _smooth_footprint(footprint: np.ndarray, smooth_kwargs: dict) -> np.ndarray:
    """Apply Gaussian smoothing to a footprint"""
    return cv2.GaussianBlur(footprint, **smooth_kwargs)


def _smooth_traces(traces: xr.DataArray, smooth_kwargs: dict) -> xr.DataArray:
    """Applies a Gaussian filter to the traces along the time axis."""
    smooth_traces = gaussian_filter1d(traces.transpose(AXIS.component_dim, ...), **smooth_kwargs)
    return xr.DataArray(smooth_traces, dims=traces.dims, coords=traces.coords)


def _gather_discrete(
    fps: xr.DataArray, trs: xr.DataArray, merge_groups: xr.DataArray | None = None
) -> tuple[list[xr.DataArray], list[xr.DataArray]]:
    """
    Gather and register new cells that are not merged to existing cells.

    """
    if merge_groups is not None:
        discrete_idx = merge_groups.where(
            merge_groups.sum(AXIS.duplicate(AXIS.component_dim)) == 0, drop=True
        )[AXIS.component_dim].values
    else:
        discrete_idx = np.arange(fps.sizes[AXIS.component_dim])

    footprints = []
    traces = []

    for idx in discrete_idx:
        reg_fps, reg_trs = _register(
            shapes=fps.isel({AXIS.component_dim: idx}),
            tracks=trs.isel({AXIS.component_dim: idx}),
        )
        footprints.append(reg_fps)
        traces.append(reg_trs)

    return footprints, traces


def _get_absorption_targets(
    existing_fp: xr.DataArray, existing_tr: xr.DataArray, age_limit: int
) -> tuple[csr_matrix | None, xr.DataArray | None]:
    if existing_fp is None or existing_tr is None:
        return None, None

    targets = existing_tr[AXIS.detect_coord].values > (
        existing_tr[AXIS.frame_coord].values.max() - age_limit
    )
    known_fp = existing_fp.data.reshape((existing_fp.sizes[AXIS.component_dim], -1)).tocsr()[
        targets
    ]
    known_tr = existing_tr[targets]

    return known_fp, known_tr


def _register(shapes: xr.DataArray, tracks: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Give appropriate coordinates and assign id and detected_on to component
    shape and trace matrix

    """
    if AXIS.component_dim not in shapes.dims:
        shapes = shapes.expand_dims(AXIS.component_dim)
        tracks = tracks.expand_dims(AXIS.component_dim)

    count = shapes.sizes[AXIS.component_dim]
    new_ids = [create_id() for _ in range(count)]

    coords = {
        AXIS.id_coord: (AXIS.component_dim, new_ids),
        AXIS.detect_coord: (
            AXIS.component_dim,
            [tracks[AXIS.frame_coord].max().item()] * count,
        ),
    }

    footprints = shapes.assign_coords(coords)
    traces = tracks.assign_coords(coords)

    return footprints, traces


def _create_component_movie(
    footprint: xr.DataArray | csr_matrix,  # Can be xr.DataArray or csr_matrix
    trace: xr.DataArray,
) -> np.ndarray:
    """Movie of a single component"""
    clean_trace = trace.dropna(dim=AXIS.frame_dim).data

    if isinstance(footprint, csr_matrix):
        # Target (CSR matrix) case: Transpose the footprint matrix
        movie = footprint.T @ clean_trace
    else:
        # New component (DataArray) case: Transpose the spatial dimensions
        # np.matmul works fine if dense
        movie = np.matmul(
            footprint.transpose(*AXIS.spatial_dims, ...).data,
            clean_trace,
        )
    return movie
