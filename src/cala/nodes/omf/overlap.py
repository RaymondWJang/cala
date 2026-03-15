import numpy as np

from cala.arrays import AXIS
from cala.arrays.models import Footprints, Overlaps, assemble_sparse_bool, overlap_format
from cala.util import sp_matmul, stack_sparse


def initialize(overlaps: Overlaps, footprints: Footprints) -> Overlaps:
    A = footprints.array

    if A is None:
        return overlaps

    V = sp_matmul(left=A, dim=AXIS.component_dim, rename_map=AXIS.component_rename)

    overlaps.array = V > 0

    return overlaps


def ingest_component(
    overlaps: Overlaps, footprints: Footprints, new_footprints: Footprints
) -> Overlaps:
    """Update component overlap matrix with new components.

    Updates the binary adjacency matrix that represents component overlaps.
    Matrix element (i,j) is 1 if components i and j overlap spatially, 0 otherwise.

    Args:
        footprints (Footprints): Current spatial footprints [A, b]
        new_footprints (Footprints): Newly detected spatial components
    """
    no_new = new_footprints.array is None
    if no_new:
        return overlaps

    no_overlaps = overlaps.array is None
    if no_overlaps:
        return initialize(overlaps, new_footprints)

    A = stack_sparse(footprints.array, AXIS.component_dim).tocsr()
    V = overlaps.array.data.tocsr()

    a_new = new_footprints.array

    merged_ids = a_new.attrs.get("replaces", [])
    intact_mask = ~np.isin(overlaps.array[AXIS.id_coord].values, merged_ids)
    V_side = overlaps.array[AXIS.component_dim]

    if merged_ids:
        A = A[intact_mask]
        V = V[intact_mask].T[intact_mask]  # symmetric matrix
        V_side = V_side[intact_mask]

    a_sparse = stack_sparse(a_new, AXIS.component_dim).tocsr()

    # Compute spatial overlaps between new and existing components
    v_topright = (A @ a_sparse.T).nonzero()
    v_bottleft = v_topright[::-1]

    # Compute overlaps between new components themselves
    v_botright = a_sparse @ a_sparse.T

    # Construct the new overlap matrix by blocks
    # [     V      v_topright]
    # [v_bottleft  v_botright]
    updated_overlaps = assemble_sparse_bool(
        V.nonzero(), v_topright, v_bottleft, v_botright.nonzero(), V.shape, v_botright.shape
    )

    overlaps.array = overlap_format(updated_overlaps, V_side, a_new.coords)

    return overlaps
