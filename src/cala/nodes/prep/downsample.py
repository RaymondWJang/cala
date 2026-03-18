from typing import Annotated as A

import numpy as np
from noob import Name

from cala.arrays import AXIS, Frame
from cala.nodes.prep import package_frame


def downsample(frames: list[Frame]) -> A[Frame, Name("frame")]:
    """
    Downsampling in time and cropping in space. Must be followed by gather node, and
    t_downsample has to be same as gather's parameter n value.

    :param frames:
    """
    arrays = [f.array for f in frames]
    return package_frame(
        np.mean(arrays, axis=0), arrays[-1][AXIS.frame_coord].item() // len(arrays)
    )
