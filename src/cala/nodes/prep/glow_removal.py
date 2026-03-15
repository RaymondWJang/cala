from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name, process_method

from cala.arrays import Frame


class GlowRemover:
    base_brightness_: np.ndarray = None

    @process_method
    def process(self, frame: Frame) -> A[Frame, Name("frame")]:
        frame = frame.array

        if self.base_brightness_ is None:
            self.base_brightness_ = frame.values

        self.base_brightness_ = np.minimum(frame.values, self.base_brightness_)

        shifted = (frame - self.base_brightness_).values

        return Frame.from_array(xr.DataArray(shifted, dims=frame.dims, coords=frame.coords))
