from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name, process_method
from pydantic import BaseModel
from scipy.sparse import csc_matrix, vstack
from sparse import COO

from cala.arrays import AXIS, CompStats, Footprints, PixStats
from cala.logging import init_logger
from cala.util import concatenate_coordinates


class Footprinter(BaseModel):
    max_iter: int
    ratio_lb: float
    """low bound for footprint value. values lower than this is floored to zero."""

    _logger = init_logger(__name__)

    @process_method
    def ingest_frame(
        self, footprints: Footprints, pixel_stats: PixStats, component_stats: CompStats
    ) -> A[Footprints, Name("footprints")]:
        """
        Update spatial footprints using sufficient statistics.
        We don't use tolerance since Frobenius norm is too expensive to
        calculate with the entire footprints stack.

            Ã[p, i] = max(Ã[p, i] + (W[p, i] - Ã[p, :]M[:, i])/M[i, i], 0)

        where:
            - Ã is the spatial footprints matrix
            - W is the pixel-wise sufficient statistics
            - M is the component-wise sufficient statistics
            - p are the pixels where component i can be non-zero

        Args:
            pixel_stats (PixStats): Sufficient statistics W.
                Shape: (pixels × components)
            component_stats (CompStats): Sufficient statistics M.
                Shape: (components × components)
        """
        if footprints.array is None:
            return footprints

        A = footprints.array.transpose(AXIS.component_dim, ...)
        # gotta convert to csc twice...otherwise COO tries to compile numba 😭
        A_arr = A.data.reshape((A.sizes[AXIS.component_dim], -1)).tocsc()
        M = component_stats.array
        W = pixel_stats.array.transpose(AXIS.component_dim, ...)
        W_arr = W.data.reshape((W.sizes[AXIS.component_dim], -1))

        shapes, mask, _ = update_shapes(
            CY=W_arr,
            CC=M.values,
            Ab=A_arr.T.tocsc(),
            A_mask=[Ap.nonzero()[0] for Ap in A_arr],
        )

        # maybe this happens before footprint update?
        shapes.data[shapes.data < self.ratio_lb] = 0

        footprints.array = xr.DataArray(
            COO.from_scipy_sparse(shapes.T).reshape(A.shape), dims=A.dims, coords=A.coords
        )

        return footprints


def ingest_component(
    footprints: Footprints, new_footprints: Footprints
) -> A[Footprints, Name("footprints")]:
    if new_footprints.array is None:
        return footprints

    a = footprints.array
    a_new = new_footprints.array

    if footprints.array is None:
        footprints.array = a_new
        return footprints

    a_val = a.data.reshape((a.sizes[AXIS.component_dim], -1)).tocsr()
    a_ids = a[AXIS.id_coord].values
    a_side = a[AXIS.component_dim]
    a_new_val = a_new.data.reshape((a_new.sizes[AXIS.component_dim], -1)).tocsr()

    merged_ids = a_new.attrs.get("replaces")
    if merged_ids:
        intact_mask = ~np.isin(a_ids, merged_ids)
        a_val = a_val[intact_mask]
        a_side = a_side[intact_mask]

    data = vstack([a_val, a_new_val])
    coords = concatenate_coordinates(a_side.coords, a_new[AXIS.component_dim].coords)

    footprints.array = (
        xr.DataArray(
            COO.from_scipy_sparse(data).reshape((data.shape[0], *a.shape[1:])),
            dims=a.dims,
            coords={k: (AXIS.component_dim, v) for k, v in coords.items()},
        )
        .assign_coords(a[AXIS.width_coord].coords)
        .assign_coords(a[AXIS.height_coord].coords)
    )

    return footprints


def update_shapes(
    CY: np.ndarray,
    CC: np.ndarray,
    Ab: csc_matrix,
    A_mask: list[np.ndarray],
    Ab_dense: np.ndarray | None = None,
    iters: int = 5,
) -> tuple[csc_matrix, list[np.ndarray], np.ndarray]:
    """
    :param CY: suff stats (comp, pixel)
    :param CC: suff stats (component), shape (comp, comp)
    :param Ab: shape matrix (sparse), shape (pixel, comp)
    :param A_mask: list of nonzero coordinates for each footprint list[(pixel,)]
    :param Ab_dense: shape matrix (dense)
    :param iters: number of iterations
    """
    D, M = Ab.shape

    for _ in range(iters):  # it's presumably better to run just 1 iter but update more neurons
        for m in range(M):
            tmp = _update(Ab_dense=Ab_dense, Ab=Ab, CY=CY, CC=CC, m=m, ind_pixels=A_mask[m])
            Ab_dense, Ab, A_mask = _normalize(
                m=m, Ab=Ab, Ab_dense=Ab_dense, ind_A=A_mask, ind_pixels=A_mask[m], tmp=tmp
            )

    return Ab, A_mask, Ab_dense


def _update(
    Ab_dense: np.ndarray,
    Ab: csc_matrix,
    CY: np.ndarray,
    CC: np.ndarray,
    m: int,
    ind_pixels: int,
) -> np.ndarray:
    """
    Update a single footprint

    :param Ab_dense: shape matrix (dense)
    :param Ab: shape matrix (sparse)
    :param CY: suff stats (pixel)
    :param CC: suff stats (component)
    :param m: neuron index
    :param ind_pixels: index of cell
    """
    if Ab_dense is None:
        result = np.maximum(
            Ab.data[Ab.indptr[m] : Ab.indptr[m + 1]]
            + (
                (CY[m, ind_pixels] - Ab.dot(CC[m])[ind_pixels])
                / (CC[m, m] + np.finfo(CC.dtype).eps)
            ),
            0,
        )
    else:
        result = np.maximum(
            Ab_dense[ind_pixels, m]
            + (
                (CY[m, ind_pixels] - Ab_dense[ind_pixels].dot(CC[m]))
                / (CC[m, m] + np.finfo(CC.dtype).eps)
            ),
            0,
        )

    return result


def _normalize(
    m: int,
    Ab: csc_matrix,
    Ab_dense: np.ndarray | None,
    ind_A: list[np.ndarray],
    ind_pixels: int,
    tmp: np.ndarray,
) -> tuple[np.ndarray, csc_matrix, list[int]]:
    """
    This only exists to prevent footprint values from blowing up / diminishing.
    (Hopefully) Irrelevant since we normalize footprint to the actual pixel values.

    :param m: neuron index
    :param Ab: shape matrix (sparse)
    :param Ab_dense: shape matrix (dense)
    :param ind_A: shape matrix of cells
    :param ind_pixels: shape array of a cell
    :param tmp: updated shape - before normalization
    """
    if tmp.dot(tmp) > 0:
        # tmp *= 1e-3 / min(1e-3, np.sqrt(tmp.dot(tmp)) + np.finfo(float).eps)
        if Ab_dense is not None:
            Ab_dense[ind_pixels, m] = tmp  # / max(1, np.sqrt(tmp.dot(tmp)))
            Ab.data[Ab.indptr[m] : Ab.indptr[m + 1]] = Ab_dense[ind_pixels, m]
        else:
            # tmp = tmp / max(1, np.sqrt(tmp.dot(tmp)))
            Ab.data[Ab.indptr[m] : Ab.indptr[m + 1]] = tmp
        ind_A[m] = Ab.indices[slice(Ab.indptr[m], Ab.indptr[m + 1])]

    return Ab_dense, Ab, ind_A
