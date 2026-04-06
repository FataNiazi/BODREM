from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import gymnasium as gym

from .halfcheetah_params import HalfCheetahParams, coerce_params

try:
    import mujoco
    HAS_MUJOCO = True
except Exception:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]
    HAS_MUJOCO = False


HALFCHEETAH_DEFAULT_ENV_ID = "HalfCheetah-v4"
_BODY_NAME_TO_FIELD = {
    "back_thigh_mass": "bthigh",
    "back_shin_mass": "bshin",
    "back_foot_mass": "bfoot",
    "front_thigh_mass": "fthigh",
    "front_shin_mass": "fshin",
    "front_foot_mass": "ffoot",
}
_FRICTION_GEOM_NAME = "floor"


@dataclass(slots=True)
class HalfCheetahParamApplier:
    body_name_to_field: dict[str, str] = field(default_factory=lambda: dict(_BODY_NAME_TO_FIELD))
    floor_geom_name: str = _FRICTION_GEOM_NAME

    def apply(
        self,
        env: Any,
        params: HalfCheetahParams | Mapping[str, float],
    ) -> HalfCheetahParams:
        if mujoco is None:
            raise RuntimeError(
                "HalfCheetah parameter mutation requires the 'mujoco' package. "
                "Install via 'pip install gymnasium[mujoco]' or 'pip install mujoco'."
            )

        params_obj = coerce_params(params)
        unwrapped = env.unwrapped
        model = getattr(unwrapped, "model", None)
        data = getattr(unwrapped, "data", None)
        if model is None:
            raise RuntimeError("Expected a MuJoCo-backed env with 'env.unwrapped.model'.")

        for field_name, body_name in self.body_name_to_field.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise RuntimeError(f"Could not resolve HalfCheetah body '{body_name}'.")
            model.body_mass[body_id] = float(getattr(params_obj, field_name))

        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.floor_geom_name)
        if geom_id < 0:
            raise RuntimeError(f"Could not resolve HalfCheetah geom '{self.floor_geom_name}'.")
        model.geom_friction[geom_id, 0] = float(params_obj.surface_friction)

        if data is not None:
            mujoco.mj_forward(model, data)
        return params_obj


_DEFAULT_APPLIER = HalfCheetahParamApplier()


def set_env_params(
    env: Any,
    params: HalfCheetahParams | Mapping[str, float],
    applier: HalfCheetahParamApplier | None = None,
) -> HalfCheetahParams:
    return (applier or _DEFAULT_APPLIER).apply(env, params)


def make_halfcheetah_env(
    params: HalfCheetahParams | Mapping[str, float] | None = None,
    seed: int | None = None,
    render_mode: str | None = None,
    env_id: str = HALFCHEETAH_DEFAULT_ENV_ID,
    **kwargs: Any,
):
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    if params is not None:
        set_env_params(env, params)
    env.reset(seed=seed)
    return env


__all__ = [
    "HALFCHEETAH_DEFAULT_ENV_ID",
    "HAS_MUJOCO",
    "HalfCheetahParamApplier",
    "make_halfcheetah_env",
    "set_env_params",
]
