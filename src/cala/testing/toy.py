from collections.abc import Generator, Iterable
from datetime import datetime, timedelta
from typing import Self, TypeVar

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator
from skimage.morphology import disk

from cala.arrays import AXIS, Frame, Movie
from cala.arrays.models import Footprints, Traces


class FrameDims(BaseModel):
    width: int
    height: int


class Position(BaseModel):
    width: int
    height: int


_TFrame = TypeVar("_TFrame", xr.DataArray, Frame)


class Toy(BaseModel):
    """
    Ex:

    frame_dims = FrameSize(width=1024, height=512)
    cell_positions = [Position(width=200, height=300)]
    cell_traces = [np.array(range(100))]

    toy = Toy(
        n_frames=100,
        frame_dims=frame_dims,
        cell_radii=20,
        cell_positions=cell_positions,
        cell_traces=cell_traces,
    )

    cell_position = Position(width=400, height=300)
    cell_trace = np.array(range(100, 0, -1))

    toy.add_cell(cell_position, 10, cell_trace, "cell_a")

    toy.drop_cell("cell_a")

    toy_movie = toy.make_movie()
    """

    n_frames: int
    frame_dims: FrameDims
    cell_radii: int | list[int]
    cell_positions: list[Position]
    cell_traces: list[np.ndarray]
    cell_ids: list[str] = None
    """If none, auto populated as cell_{idx}."""
    detected_ons: list[int] = None
    emit_frames: bool = False

    _footprints: xr.DataArray = PrivateAttr(init=False)
    _traces: xr.DataArray = PrivateAttr(init=False)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("n_frames", mode="after")
    @classmethod
    def natural_num(cls, value: int) -> int:
        assert value >= 0, "n_frames must be positive."
        return value

    @model_validator(mode="before")
    def fill_ids(self) -> Self:
        if self.get("cell_ids", None) is None:
            self["cell_ids"] = [f"cell_{idx}" for idx, _ in enumerate(self["cell_positions"])]
        return self

    @model_validator(mode="before")
    def fill_detected_ons(self) -> Self:
        if self.get("detected_ons", None) is None:
            self["detected_ons"] = [0] * len(self["cell_positions"])
        return self

    @model_validator(mode="before")
    def fill_radii(self) -> Self:
        self["cell_radii"] = (
            [self["cell_radii"]] * len(self["cell_positions"])
            if isinstance(self["cell_radii"], int)
            else self["cell_radii"]
        )
        return self

    @model_validator(mode="after")
    def consistent_n_cells(self) -> Self:
        for cell_trace in self.cell_traces:
            assert self.n_frames == len(
                cell_trace
            ), "inconsistent n_frames between n_frames and cell_traces"
        return self

    @model_validator(mode="after")
    def consistent_n_frames(self) -> Self:
        assert len(self.cell_positions) == len(self.cell_traces), (
            f"inconsistent cell counts. "
            f"positions: {len(self.cell_positions)}, "
            f"traces: {len(self.cell_traces)}"
        )
        return self

    @model_validator(mode="after")
    def cells_within_bounds(self) -> Self:
        for position, radius in zip(self.cell_positions, self.cell_radii):
            assert np.min([position.width, position.height]) - radius > 0
            assert position.width + radius < self.frame_dims.width
            assert position.height + radius < self.frame_dims.height
        return self

    def model_post_init(self, __context: None = None) -> None:
        self._footprints = self._build_footprints()
        self._traces = self._build_traces()

    @property
    def n_components(self) -> int:
        return len(self.cell_positions)

    def _build_movie_template(self) -> xr.DataArray:
        return xr.DataArray(
            np.zeros((self.n_frames, self.frame_dims.height, self.frame_dims.width)),
            dims=[AXIS.frame_dim, *AXIS.spatial_dims],
        )

    def _generate_footprint(
        self, radius: int, position: Position, id_: str, detected_on: int
    ) -> xr.DataArray:
        footprint = xr.DataArray(
            np.zeros((self.frame_dims.height, self.frame_dims.width)),
            dims=AXIS.spatial_dims,
        )

        shape = disk(radius)
        shape[:radius, :radius] *= 2
        shape[radius:, :radius] *= 3
        shape[radius:, radius:] *= 4

        width_slice = slice(position.width - radius, position.width + radius + 1)
        height_slice = slice(position.height - radius, position.height + radius + 1)

        footprint.loc[{AXIS.height_dim: height_slice, AXIS.width_dim: width_slice}] = shape

        return footprint.expand_dims(AXIS.component_dim).assign_coords(
            {
                AXIS.id_coord: (AXIS.component_dim, [id_]),
                AXIS.detect_coord: (AXIS.component_dim, [detected_on]),
                AXIS.width_coord: (AXIS.width_dim, footprint[AXIS.width_dim].data),
                AXIS.height_coord: (AXIS.height_dim, footprint[AXIS.height_dim].data),
            }
        )

    def _build_footprints(self) -> xr.DataArray:
        footprints = []
        for radius, position, id_, confid in zip(
            self.cell_radii, self.cell_positions, self.cell_ids, self.detected_ons
        ):
            footprints.append(self._generate_footprint(radius, position, id_, confid))

        return xr.concat(footprints, dim=AXIS.component_dim)

    def _format_trace(self, trace: np.ndarray, id_: str, detected_on: int) -> xr.DataArray:
        return (
            xr.DataArray(
                trace,
                dims=AXIS.frame_dim,
            )
            .expand_dims(AXIS.component_dim)
            .assign_coords(
                {
                    AXIS.id_coord: (AXIS.component_dim, [id_]),
                    AXIS.detect_coord: (AXIS.component_dim, [detected_on]),
                    AXIS.frame_coord: (AXIS.frame_dim, range(trace.size)),
                }
            )
        )

    def _build_traces(self) -> xr.DataArray:
        traces = []
        for trace, id_, confid in zip(self.cell_traces, self.cell_ids, self.detected_ons):
            traces.append(self._format_trace(trace, id_, confid))

        return xr.concat(traces, dim=AXIS.component_dim).assign_coords(
            {
                AXIS.timestamp_coord: (
                    AXIS.frame_dim,
                    [
                        (datetime.now() + i * timedelta(microseconds=20)).strftime("%H:%M:%S.%f")
                        for i in range(self.n_frames)
                    ],
                )
            }
        )

    def _build_movie(self, footprints: xr.DataArray, traces: xr.DataArray) -> xr.DataArray:
        movie = self._build_movie_template()
        movie += (footprints @ traces).transpose(AXIS.frame_dim, *AXIS.spatial_dims).as_numpy()
        return movie

    def make_movie(self) -> Movie:
        movie = self._build_movie(self._footprints, self._traces)
        return Movie.from_array(movie)

    def add_cell(
        self, position: Position, radius: int, trace: np.ndarray, id_: str, detected_on: int = 0
    ) -> None:
        new_footprint = self._generate_footprint(radius, position, id_, detected_on)
        self._footprints = xr.concat([self._footprints, new_footprint], dim=AXIS.component_dim)

        new_trace = self._format_trace(trace, id_, detected_on)
        self._traces = xr.concat([self._traces, new_trace], dim=AXIS.component_dim)

    def drop_cell(self, id_: str | Iterable[str]) -> None:
        id_ = {id_} if isinstance(id_, str) else set(id_)

        id_coords = set(self._footprints.coords[AXIS.id_coord].values.tolist())

        keep_ids = list(id_coords - id_)

        self._footprints = (
            self._footprints.set_xindex(AXIS.id_coord)
            .sel({AXIS.id_coord: keep_ids})
            .reset_index(AXIS.id_coord)
        )
        self._traces = (
            self._traces.set_xindex(AXIS.id_coord)
            .sel({AXIS.id_coord: keep_ids})
            .reset_index(AXIS.id_coord)
        )

    @property
    def footprints(self) -> Footprints:
        if self._footprints.any():
            return Footprints.from_array(self._footprints)
        else:
            raise ValueError("No footprints available")

    @property
    def traces(self) -> Traces:
        if self._traces.any():
            return Traces.from_array(self._traces)
        else:
            raise ValueError("No traces available")

    def movie_gen(self) -> Generator[_TFrame]:
        for i in range(self._traces.sizes[AXIS.frame_dim]):
            trace = self._traces.isel({AXIS.frame_dim: i})
            if not self.emit_frames:
                yield trace @ self._footprints
            else:
                yield Frame.from_array(trace @ self._footprints)
