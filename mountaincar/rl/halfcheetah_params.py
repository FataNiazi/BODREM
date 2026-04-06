from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import numpy as np


HALFCHEETAH_PARAM_NAMES = (
    "back_thigh_mass",
    "back_shin_mass",
    "back_foot_mass",
    "front_thigh_mass",
    "front_shin_mass",
    "front_foot_mass",
    "surface_friction",
)


@dataclass(frozen=True)
class HalfCheetahParams:
    back_thigh_mass: float
    back_shin_mass: float
    back_foot_mass: float
    front_thigh_mass: float
    front_shin_mass: float
    front_foot_mass: float
    surface_friction: float

    def as_dict(self) -> dict[str, float]:
        return OrderedDict(
            (
                ("back_thigh_mass", float(self.back_thigh_mass)),
                ("back_shin_mass", float(self.back_shin_mass)),
                ("back_foot_mass", float(self.back_foot_mass)),
                ("front_thigh_mass", float(self.front_thigh_mass)),
                ("front_shin_mass", float(self.front_shin_mass)),
                ("front_foot_mass", float(self.front_foot_mass)),
                ("surface_friction", float(self.surface_friction)),
            )
        )

    def to_ordered_array(self) -> np.ndarray:
        return np.asarray(list(self.as_dict().values()), dtype=np.float64)

    @classmethod
    def from_array(cls, values: np.ndarray | list[float] | tuple[float, ...]) -> "HalfCheetahParams":
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size != len(HALFCHEETAH_PARAM_NAMES):
            raise ValueError(
                f"Expected {len(HALFCHEETAH_PARAM_NAMES)} HalfCheetah params, got {arr.size}."
            )
        return cls(*[float(x) for x in arr.tolist()])

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "HalfCheetahParams":
        return cls(**{name: float(values[name]) for name in HALFCHEETAH_PARAM_NAMES})


@dataclass(frozen=True)
class HalfCheetahBounds:
    back_thigh_mass: tuple[float, float] = (0.08, 2.99)
    back_shin_mass: tuple[float, float] = (0.08, 3.08)
    back_foot_mass: tuple[float, float] = (0.05, 2.08)
    front_thigh_mass: tuple[float, float] = (0.07, 2.78)
    front_shin_mass: tuple[float, float] = (0.06, 2.30)
    front_foot_mass: tuple[float, float] = (0.04, 1.66)
    surface_friction: tuple[float, float] = (0.02, 0.78)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return OrderedDict(
            (
                ("back_thigh_mass", (float(self.back_thigh_mass[0]), float(self.back_thigh_mass[1]))),
                ("back_shin_mass", (float(self.back_shin_mass[0]), float(self.back_shin_mass[1]))),
                ("back_foot_mass", (float(self.back_foot_mass[0]), float(self.back_foot_mass[1]))),
                ("front_thigh_mass", (float(self.front_thigh_mass[0]), float(self.front_thigh_mass[1]))),
                ("front_shin_mass", (float(self.front_shin_mass[0]), float(self.front_shin_mass[1]))),
                ("front_foot_mass", (float(self.front_foot_mass[0]), float(self.front_foot_mass[1]))),
                ("surface_friction", (float(self.surface_friction[0]), float(self.surface_friction[1]))),
            )
        )

    def low(self) -> np.ndarray:
        return np.asarray([low for low, _ in self.as_dict().values()], dtype=np.float64)

    def high(self) -> np.ndarray:
        return np.asarray([high for _, high in self.as_dict().values()], dtype=np.float64)


DEFAULT_HALFCHEETAH_BOUNDS = HalfCheetahBounds()


def clip_params(
    params: HalfCheetahParams | Mapping[str, float] | np.ndarray | list[float] | tuple[float, ...],
    bounds: HalfCheetahBounds | Mapping[str, tuple[float, float]] = DEFAULT_HALFCHEETAH_BOUNDS,
) -> HalfCheetahParams:
    params_obj = params if isinstance(params, HalfCheetahParams) else coerce_params(params)
    bounds_dict = bounds.as_dict() if isinstance(bounds, HalfCheetahBounds) else OrderedDict(bounds)
    clipped = []
    for name, value in params_obj.as_dict().items():
        low, high = bounds_dict[name]
        clipped.append(float(np.clip(value, low, high)))
    return HalfCheetahParams.from_array(clipped)


def sample_params(
    rng: np.random.Generator,
    bounds: HalfCheetahBounds | Mapping[str, tuple[float, float]] = DEFAULT_HALFCHEETAH_BOUNDS,
) -> HalfCheetahParams:
    bounds_dict = bounds.as_dict() if isinstance(bounds, HalfCheetahBounds) else OrderedDict(bounds)
    values = [float(rng.uniform(low, high)) for low, high in bounds_dict.values()]
    return HalfCheetahParams.from_array(values)


def coerce_params(
    params: HalfCheetahParams | Mapping[str, float] | np.ndarray | list[float] | tuple[float, ...],
) -> HalfCheetahParams:
    if isinstance(params, HalfCheetahParams):
        return params
    if isinstance(params, Mapping):
        return HalfCheetahParams.from_mapping(params)
    return HalfCheetahParams.from_array(params)
