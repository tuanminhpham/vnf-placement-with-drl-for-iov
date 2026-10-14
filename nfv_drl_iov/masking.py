from __future__ import annotations

from typing import Mapping

import numpy as np

from .env import DPVEnv


MASK_MODE_STRICT = "strict"
MASK_MODE_NONE = "none"
MASK_MODES = (MASK_MODE_STRICT, MASK_MODE_NONE)


def full_action_mask(env: DPVEnv) -> np.ndarray:
    if env.done:
        return np.zeros(env.action_space.n, dtype=bool)
    return np.ones(env.action_space.n, dtype=bool)


def policy_action_mask(
    env: DPVEnv,
    mask_mode: str,
    info: Mapping[str, object] | None = None,
) -> np.ndarray:
    if mask_mode == MASK_MODE_NONE:
        return full_action_mask(env)
    if mask_mode != MASK_MODE_STRICT:
        raise ValueError(f"unknown mask mode: {mask_mode}")
    if info is not None and "action_mask" in info:
        return np.asarray(info["action_mask"], dtype=bool)
    return env.action_masks()


def checkpoint_mask_mode(checkpoint: Mapping[str, object]) -> str:
    mask_mode = str(checkpoint.get("mask_mode", MASK_MODE_STRICT)).lower()
    return mask_mode if mask_mode in MASK_MODES else MASK_MODE_STRICT


def algorithm_name(base_name: str, mask_mode: str) -> str:
    prefix = "masked" if mask_mode == MASK_MODE_STRICT else "unmasked"
    return f"{prefix}_{base_name}"
