from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .formation_env import FormationAviaryEnv

# qp_solver signature:
#   (cbf_state[N,6], v_des[N,3], last_info) -> (v_safe[N,3], qp_info)
QPSolver = Callable[[np.ndarray, np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]]


@dataclass
class RLCBFWrapperConfig:
    enable_qp: bool = True
    clip_velocity: bool = True


class RLCBFQPWrapper(gym.Wrapper):
    """Standardized interface layer for RL + CBF-QP integration.

    Exposed action space is per-drone desired velocity in m/s, shape (N, 3).
    Internally, it converts velocity commands to the base env normalized action.
    """

    def __init__(
        self,
        env: FormationAviaryEnv,
        qp_solver: Optional[QPSolver] = None,
        config: Optional[RLCBFWrapperConfig] = None,
    ):
        super().__init__(env)
        self.cfg = config or RLCBFWrapperConfig()
        self.qp_solver = qp_solver

        self._num_drones = env.cfg.num_drones
        self._vel_low = np.array(
            [-env.cfg.max_target_speed_xy, -env.cfg.max_target_speed_xy, -env.cfg.max_target_speed_z],
            dtype=np.float32,
        )
        self._vel_high = np.array(
            [env.cfg.max_target_speed_xy, env.cfg.max_target_speed_xy, env.cfg.max_target_speed_z],
            dtype=np.float32,
        )
        # PPO-facing action space (absolute target velocity in m/s).
        self.action_space = spaces.Box(
            low=np.tile(self._vel_low, (self._num_drones, 1)),
            high=np.tile(self._vel_high, (self._num_drones, 1)),
            dtype=np.float32,
        )
        self.observation_space = env.observation_space

        self._last_info: Dict[str, Any] = {}
        self._last_cbf_state = np.zeros((self._num_drones, 6), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_info = dict(info)
        self._last_cbf_state = np.asarray(
            info.get("cbf_state", np.zeros((self._num_drones, 6), dtype=np.float32)),
            dtype=np.float32,
        )
        return obs, self._augment_info(info, None, None, None, {"status": "reset"})

    def step(self, action: np.ndarray):
        v_des = np.asarray(action, dtype=np.float32)
        if v_des.shape != (self._num_drones, 3):
            raise ValueError(f"action shape must be {(self._num_drones, 3)}, got {v_des.shape}")

        if self.cfg.clip_velocity:
            v_des = np.clip(v_des, self._vel_low[None, :], self._vel_high[None, :])

        v_safe = v_des.copy()
        qp_info: Dict[str, Any] = {"status": "bypass"}

        if self.cfg.enable_qp and self.qp_solver is not None:
            v_qp, qp_info = self.qp_solver(self._last_cbf_state.copy(), v_des.copy(), dict(self._last_info))
            v_safe = np.asarray(v_qp, dtype=np.float32)
            if v_safe.shape != (self._num_drones, 3):
                raise RuntimeError(
                    f"qp_solver must return velocity shape {(self._num_drones, 3)}, got {v_safe.shape}"
                )
            if self.cfg.clip_velocity:
                v_safe = np.clip(v_safe, self._vel_low[None, :], self._vel_high[None, :])
            qp_info = qp_info or {"status": "ok"}

        norm_action = self._velocity_to_normalized(v_safe)
        obs, reward, terminated, truncated, info = self.env.step(norm_action)

        self._last_info = dict(info)
        self._last_cbf_state = np.asarray(
            info.get("cbf_state", np.zeros((self._num_drones, 6), dtype=np.float32)),
            dtype=np.float32,
        )

        out_info = self._augment_info(info, v_des, v_safe, norm_action, qp_info)
        return obs, reward, terminated, truncated, out_info

    def set_qp_solver(self, qp_solver: Optional[QPSolver]) -> None:
        self.qp_solver = qp_solver

    def _velocity_to_normalized(self, velocity: np.ndarray) -> np.ndarray:
        return self.env.velocity_to_normalized_action(velocity)

    @staticmethod
    def _to_numpy_or_none(x: Any) -> Any:
        if x is None:
            return None
        return np.asarray(x, dtype=np.float32)

    def _augment_info(
        self,
        info: Dict[str, Any],
        v_des: Optional[np.ndarray],
        v_safe: Optional[np.ndarray],
        norm_action: Optional[np.ndarray],
        qp_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        out = dict(info)
        out["v_des"] = self._to_numpy_or_none(v_des)
        out["v_safe"] = self._to_numpy_or_none(v_safe)
        out["normalized_action"] = self._to_numpy_or_none(norm_action)
        out["qp_info"] = qp_info or {"status": "none"}
        return out
