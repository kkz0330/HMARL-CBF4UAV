from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class FormationEnvConfig:
    num_drones: int = 3
    episode_len_sec: float = 12.0
    pyb_freq: int = 240
    ctrl_freq: int = 48
    gui: bool = True
    init_height: float = 1.0
    desired_height: float = 1.0
    formation_pattern: str = "line"  # options: line, square, auto
    formation_spacing: float = 0.3
    safe_distance: float = 0.22
    collision_distance: float = 0.10
    max_target_speed_xy: float = 1.0
    max_target_speed_z: float = 0.6
    target_yaw: float = 0.0
    max_rpm: float = 22000.0
    use_moving_goal: bool = False
    goal_start_x: float = 0.0
    goal_start_y: float = 0.0
    goal_start_z: float = 1.0
    goal_end_x: float = 0.0
    goal_end_y: float = 0.0
    goal_end_z: float = 1.0
    goal_speed: float = 0.5
    goal_arrival_tolerance: float = 0.18
    goal_arrival_bonus: float = 10.0
    goal_vel_track_weight: float = 0.05
    fall_z_threshold: float = 0.15
    fall_penalty: float = 10.0
    terminate_on_fall: bool = True


class FormationAviaryEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: FormationEnvConfig | None = None):
        super().__init__()
        self.cfg = config or FormationEnvConfig()
        self._sim_env = None
        self._pid_controllers = None
        self._episode_step = 0
        self._last_raw_obs = None
        self._goal_arrival_bonus_given = False
        self._goal_arrival_event_last = False
        self._goal_debug_line_ids = [-1, -1, -1]
        self._goal_debug_text_id = -1

        self._goal_center = np.array(
            [self.cfg.goal_start_x, self.cfg.goal_start_y, self.cfg.goal_start_z],
            dtype=np.float32,
        )
        self._formation_offsets = self._build_formation_offsets(
            self.cfg.num_drones,
            spacing=self.cfg.formation_spacing,
            pattern=self.cfg.formation_pattern,
        )

        obs_dim = self.cfg.num_drones * 9 + self._pair_count(self.cfg.num_drones)
        action_low = np.full((self.cfg.num_drones, 3), -1.0, dtype=np.float32)
        action_high = np.full((self.cfg.num_drones, 3), 1.0, dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32,
        )
        obs_low = np.full((obs_dim,), -np.finfo(np.float32).max, dtype=np.float32)
        obs_high = np.full((obs_dim,), np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

    def _build_sim(self) -> None:
        if self._sim_env is not None:
            return
        try:
            from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
            from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
            from gym_pybullet_drones.utils.enums import DroneModel, Physics
        except ImportError as exc:
            raise ImportError(
                "Missing dependency gym-pybullet-drones. Install dependencies first."
            ) from exc

        init_xyzs = np.zeros((self.cfg.num_drones, 3), dtype=np.float32)
        init_xyzs[:, 0] = self._formation_offsets[:, 0]
        init_xyzs[:, 1] = self._formation_offsets[:, 1]
        init_xyzs[:, 2] = self.cfg.init_height

        self._sim_env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.cfg.num_drones,
            initial_xyzs=init_xyzs,
            initial_rpys=np.zeros((self.cfg.num_drones, 3), dtype=np.float32),
            physics=Physics.PYB,
            neighbourhood_radius=3.0,
            pyb_freq=self.cfg.pyb_freq,
            ctrl_freq=self.cfg.ctrl_freq,
            gui=self.cfg.gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        self._pid_controllers = [
            DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(self.cfg.num_drones)
        ]

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._build_sim()
        self._episode_step = 0
        self._goal_arrival_bonus_given = False
        self._goal_arrival_event_last = False
        self._update_goal_center()

        reset_out = self._sim_env.reset(seed=seed)
        raw_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self._last_raw_obs = raw_obs
        self._update_goal_debug_marker()
        for ctrl in self._pid_controllers:
            ctrl.reset()
        obs = self._build_obs(raw_obs)
        info = self._build_info(
            raw_obs,
            np.zeros((self.cfg.num_drones, 3), dtype=np.float32),
            np.zeros((self.cfg.num_drones, 3), dtype=np.float32),
        )
        return obs, info

    def step(self, action: np.ndarray):
        self._episode_step += 1
        self._update_goal_center()
        self._update_goal_debug_marker()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.cfg.num_drones, 3):
            raise ValueError(f"action shape must be {(self.cfg.num_drones, 3)}, got {action.shape}")

        rpm, target_vel = self._action_to_rpm_with_pid(action, self._last_raw_obs)
        step_out = self._sim_env.step(rpm)

        if len(step_out) == 5:
            raw_obs, _, terminated, truncated, _ = step_out
        elif len(step_out) == 4:
            raw_obs, _, done, _ = step_out
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError("Unexpected CtrlAviary step() output format.")
        self._last_raw_obs = raw_obs

        obs = self._build_obs(raw_obs)
        reward = self._compute_reward(raw_obs, action)
        info = self._build_info(raw_obs, action, target_vel)

        if self._episode_step >= int(self.cfg.episode_len_sec * self.cfg.ctrl_freq):
            truncated = True
        if info["min_pairwise_distance"] < self.cfg.collision_distance:
            terminated = True
        if self.cfg.terminate_on_fall and bool(info.get("fall_event", False)):
            terminated = True

        return obs, float(reward), bool(terminated), bool(truncated), info

    def step_velocity(self, velocity: np.ndarray):
        """Step with per-drone target velocity command in m/s, shape (N, 3)."""
        norm_action = self.velocity_to_normalized_action(velocity)
        return self.step(norm_action)

    def close(self):
        if self._sim_env is not None:
            self._sim_env.close()
            self._sim_env = None
        self._pid_controllers = None
        self._last_raw_obs = None
        self._goal_debug_line_ids = [-1, -1, -1]
        self._goal_debug_text_id = -1

    def normalized_action_to_velocity(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.cfg.num_drones, 3):
            raise ValueError(f"normalized action shape must be {(self.cfg.num_drones, 3)}, got {action.shape}")
        clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_vel = np.zeros((self.cfg.num_drones, 3), dtype=np.float32)
        target_vel[:, 0:2] = clipped[:, 0:2] * self.cfg.max_target_speed_xy
        target_vel[:, 2] = clipped[:, 2] * self.cfg.max_target_speed_z
        return target_vel

    def velocity_to_normalized_action(self, velocity: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity, dtype=np.float32)
        if velocity.shape != (self.cfg.num_drones, 3):
            raise ValueError(f"velocity shape must be {(self.cfg.num_drones, 3)}, got {velocity.shape}")
        action = np.zeros_like(velocity, dtype=np.float32)
        action[:, 0:2] = velocity[:, 0:2] / float(self.cfg.max_target_speed_xy)
        action[:, 2] = velocity[:, 2] / float(self.cfg.max_target_speed_z)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _action_to_target_vel(self, action: np.ndarray) -> np.ndarray:
        return self.normalized_action_to_velocity(action)

    def _action_to_rpm_with_pid(self, action: np.ndarray, raw_obs: Any) -> Tuple[np.ndarray, np.ndarray]:
        target_vel = self._action_to_target_vel(action)
        rpms = np.zeros((self.cfg.num_drones, 4), dtype=np.float32)
        target_rpy = np.array([0.0, 0.0, self.cfg.target_yaw], dtype=np.float32)
        target_rpy_rates = np.zeros(3, dtype=np.float32)
        dt = 1.0 / float(self.cfg.ctrl_freq)

        for i in range(self.cfg.num_drones):
            state = np.asarray(self._get_single_drone_state(raw_obs, i), dtype=np.float32).reshape(-1)
            if state.size < 16:
                raise RuntimeError(f"State length too short for PID control on drone {i}: {state.size}")

            cur_pos = state[0:3]
            cur_quat = state[3:7]
            cur_vel = state[10:13]
            cur_ang_vel = state[13:16]
            # Velocity-tracking mode: keep current position reference and let PID track target_vel.
            target_pos = cur_pos.copy()

            rpm, _, _ = self._pid_controllers[i].computeControl(
                control_timestep=dt,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_ang_vel,
                target_pos=target_pos,
                target_rpy=target_rpy,
                target_vel=target_vel[i],
                target_rpy_rates=target_rpy_rates,
            )
            rpms[i] = np.asarray(rpm, dtype=np.float32)

        return np.clip(rpms, 0.0, self.cfg.max_rpm).astype(np.float32), target_vel

    def _current_desired_positions(self) -> np.ndarray:
        return self._goal_center[None, :] + self._formation_offsets

    def _current_goal_velocity(self) -> np.ndarray:
        """Current virtual leader velocity used by reward shaping."""
        if not self.cfg.use_moving_goal:
            return np.zeros((3,), dtype=np.float32)

        start = np.array(
            [self.cfg.goal_start_x, self.cfg.goal_start_y, self.cfg.goal_start_z],
            dtype=np.float32,
        )
        end = np.array(
            [self.cfg.goal_end_x, self.cfg.goal_end_y, self.cfg.goal_end_z],
            dtype=np.float32,
        )
        delta = end - start
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-8:
            return np.zeros((3,), dtype=np.float32)
        unit_dir = delta / dist

        speed = float(max(self.cfg.goal_speed, 0.0))
        if speed <= 1e-8:
            total_time = max(float(self.cfg.episode_len_sec), 1e-6)
            if self._episode_step >= int(self.cfg.episode_len_sec * self.cfg.ctrl_freq):
                return np.zeros((3,), dtype=np.float32)
            return (delta / total_time).astype(np.float32)

        if self._is_goal_halted():
            return np.zeros((3,), dtype=np.float32)
        return (unit_dir * speed).astype(np.float32)

    def _update_goal_center(self) -> None:
        if not self.cfg.use_moving_goal:
            if self._episode_step == 0:
                self._goal_center[:] = np.array(
                    [self.cfg.goal_start_x, self.cfg.goal_start_y, self.cfg.goal_start_z],
                    dtype=np.float32,
                )
            return

        start = np.array(
            [self.cfg.goal_start_x, self.cfg.goal_start_y, self.cfg.goal_start_z],
            dtype=np.float32,
        )
        end = np.array(
            [self.cfg.goal_end_x, self.cfg.goal_end_y, self.cfg.goal_end_z],
            dtype=np.float32,
        )
        delta = end - start
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-8:
            self._goal_center[:] = end
            return

        speed = float(max(self.cfg.goal_speed, 0.0))
        if speed <= 1e-8:
            total_steps = max(int(self.cfg.episode_len_sec * self.cfg.ctrl_freq), 1)
            tau = float(np.clip(self._episode_step / float(total_steps), 0.0, 1.0))
            self._goal_center[:] = (1.0 - tau) * start + tau * end
            return

        t_sec = float(self._episode_step) / float(max(self.cfg.ctrl_freq, 1))
        travel = min(speed * t_sec, dist)
        if travel >= dist - 1e-8:
            # Clamp to exact endpoint and hold there for the remainder of the episode.
            self._goal_center[:] = end
            return

        self._goal_center[:] = start + (travel / dist) * delta

    def _is_goal_halted(self) -> bool:
        if not self.cfg.use_moving_goal:
            return False
        end = np.array(
            [self.cfg.goal_end_x, self.cfg.goal_end_y, self.cfg.goal_end_z],
            dtype=np.float32,
        )
        return bool(np.linalg.norm(self._goal_center - end) <= 1e-3)

    def _update_goal_debug_marker(self) -> None:
        if not self.cfg.gui or self._sim_env is None:
            return
        try:
            import pybullet as p
        except ImportError:
            return

        client_id = getattr(self._sim_env, "CLIENT", None)
        if client_id is None:
            return

        c = np.asarray(self._goal_center, dtype=np.float32)
        s = 0.12
        ids = list(self._goal_debug_line_ids)
        while len(ids) < 3:
            ids.append(-1)
        self._goal_debug_line_ids = [
            p.addUserDebugLine(
                [float(c[0] - s), float(c[1]), float(c[2])],
                [float(c[0] + s), float(c[1]), float(c[2])],
                [1.0, 0.2, 0.2],
                2.0,
                0.0,
                replaceItemUniqueId=int(ids[0]),
                physicsClientId=client_id,
            ),
            p.addUserDebugLine(
                [float(c[0]), float(c[1] - s), float(c[2])],
                [float(c[0]), float(c[1] + s), float(c[2])],
                [0.2, 1.0, 0.2],
                2.0,
                0.0,
                replaceItemUniqueId=int(ids[1]),
                physicsClientId=client_id,
            ),
            p.addUserDebugLine(
                [float(c[0]), float(c[1]), float(c[2] - s)],
                [float(c[0]), float(c[1]), float(c[2] + s)],
                [0.2, 0.7, 1.0],
                2.0,
                0.0,
                replaceItemUniqueId=int(ids[2]),
                physicsClientId=client_id,
            ),
        ]
        label = "vgoal(stop)" if self._is_goal_halted() else "vgoal(move)"
        self._goal_debug_text_id = p.addUserDebugText(
            label,
            [float(c[0]), float(c[1]), float(c[2] + 0.18)],
            [1.0, 0.95, 0.2],
            1.2,
            0.0,
            replaceItemUniqueId=int(self._goal_debug_text_id),
            physicsClientId=client_id,
        )

    def _build_obs(self, raw_obs: Any) -> np.ndarray:
        pos, vel = self._parse_pos_vel(raw_obs)
        desired = self._current_desired_positions()
        err = desired - pos
        pairwise = self._pairwise_distances(pos)
        return np.concatenate([pos.reshape(-1), vel.reshape(-1), err.reshape(-1), pairwise], axis=0).astype(np.float32)

    def _compute_reward(self, raw_obs: Any, action: np.ndarray) -> float:
        pos, vel = self._parse_pos_vel(raw_obs)
        desired = self._current_desired_positions()
        goal_vel = self._current_goal_velocity()
        pairwise = self._pairwise_distances(pos)
        min_altitude = float(np.min(pos[:, 2])) if pos.shape[0] > 0 else float("inf")

        track_err = np.linalg.norm(pos - desired, axis=1)
        track_cost = float(np.mean(track_err))
        vel_track_cost = float(np.mean(np.linalg.norm(vel - goal_vel[None, :], axis=1)))
        act_cost = np.mean(action ** 2)
        safe_cost = float(np.sum(np.maximum(0.0, self.cfg.safe_distance - pairwise)))
        fall_event = 1.0 if min_altitude < float(self.cfg.fall_z_threshold) else 0.0
        self._goal_arrival_event_last = False
        arrival_bonus = 0.0
        if (
            self.cfg.use_moving_goal
            and (not self._goal_arrival_bonus_given)
            and self._is_goal_halted()
            and float(np.max(track_err)) <= float(self.cfg.goal_arrival_tolerance)
        ):
            arrival_bonus = float(self.cfg.goal_arrival_bonus)
            self._goal_arrival_bonus_given = True
            self._goal_arrival_event_last = True

        return (
            -2.0 * track_cost
            - float(self.cfg.goal_vel_track_weight) * vel_track_cost
            - 0.02 * act_cost
            - 2.0 * safe_cost
            - float(self.cfg.fall_penalty) * fall_event
            + arrival_bonus
        )

    def _build_info(self, raw_obs: Any, action: np.ndarray, target_vel: np.ndarray) -> Dict[str, Any]:
        pos, vel = self._parse_pos_vel(raw_obs)
        desired = self._current_desired_positions()
        goal_vel = self._current_goal_velocity()
        pairwise = self._pairwise_distances(pos)
        min_dist = float(np.min(pairwise)) if pairwise.size > 0 else float("inf")
        min_altitude = float(np.min(pos[:, 2])) if pos.shape[0] > 0 else float("inf")
        fall_event = bool(min_altitude < float(self.cfg.fall_z_threshold))

        return {
            "desired_positions": desired,
            "goal_center": self._goal_center.copy(),
            "goal_velocity": goal_vel.copy(),
            "goal_halted": bool(self._is_goal_halted()),
            "goal_arrival_event": bool(self._goal_arrival_event_last),
            "goal_arrival_bonus_given": bool(self._goal_arrival_bonus_given),
            "mean_track_error": float(np.mean(np.linalg.norm(pos - desired, axis=1))),
            "mean_goal_vel_track_error": float(np.mean(np.linalg.norm(vel - goal_vel[None, :], axis=1))),
            "min_pairwise_distance": min_dist,
            "cbf_margin": float(min_dist - self.cfg.safe_distance),
            "min_altitude": min_altitude,
            "fall_event": fall_event,
            "cbf_state": self.get_cbf_state(raw_obs),
            "target_velocity": target_vel.astype(np.float32),
            "action_l2": float(np.mean(action ** 2)),
        }

    def get_cbf_state(self, raw_obs: Any) -> np.ndarray:
        """Return CBF state with shape (N, 6): [x, y, z, vx, vy, vz]."""
        pos, vel = self._parse_pos_vel(raw_obs)
        return np.concatenate([pos, vel], axis=1).astype(np.float32)

    def _parse_pos_vel(self, raw_obs: Any) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.zeros((self.cfg.num_drones, 3), dtype=np.float32)
        vel = np.zeros((self.cfg.num_drones, 3), dtype=np.float32)

        for i in range(self.cfg.num_drones):
            state = self._get_single_drone_state(raw_obs, i)
            flat = np.asarray(state, dtype=np.float32).reshape(-1)
            if flat.size < 13:
                raise RuntimeError(f"State length too short for drone {i}: {flat.size}")
            pos[i] = flat[0:3]
            vel[i] = flat[10:13]
        return pos, vel

    def _get_single_drone_state(self, raw_obs: Any, i: int) -> np.ndarray:
        if isinstance(raw_obs, dict):
            if str(i) in raw_obs:
                return raw_obs[str(i)]
            if i in raw_obs:
                return raw_obs[i]
            if "state" in raw_obs:
                return raw_obs["state"][i]
        arr = np.asarray(raw_obs)
        if arr.ndim == 2:
            return arr[i]
        if arr.ndim == 1 and arr.size % self.cfg.num_drones == 0:
            stride = arr.size // self.cfg.num_drones
            return arr[i * stride : (i + 1) * stride]
        raise RuntimeError("Cannot parse CtrlAviary observation format.")

    @staticmethod
    def _pair_count(n: int) -> int:
        return n * (n - 1) // 2

    @classmethod
    def _build_formation_offsets(cls, num_drones: int, spacing: float, pattern: str) -> np.ndarray:
        p = str(pattern).lower().strip()
        if p == "auto":
            side = int(np.sqrt(max(num_drones, 1)))
            p = "square" if side * side == num_drones else "line"
        if p == "square":
            return cls._build_square_formation_offsets(num_drones, spacing)
        return cls._build_line_formation_offsets(num_drones, spacing)

    @staticmethod
    def _build_line_formation_offsets(num_drones: int, spacing: float) -> np.ndarray:
        center = 0.5 * (num_drones - 1)
        offsets = np.zeros((num_drones, 3), dtype=np.float32)
        for i in range(num_drones):
            offsets[i, 1] = (i - center) * spacing
        return offsets

    @staticmethod
    def _build_square_formation_offsets(num_drones: int, spacing: float) -> np.ndarray:
        side = int(np.sqrt(max(num_drones, 1)))
        if side * side != num_drones:
            return FormationAviaryEnv._build_line_formation_offsets(num_drones, spacing)
        center = 0.5 * (side - 1)
        offsets = np.zeros((num_drones, 3), dtype=np.float32)
        idx = 0
        for ix in range(side):
            for iy in range(side):
                offsets[idx, 0] = (ix - center) * spacing
                offsets[idx, 1] = (iy - center) * spacing
                idx += 1
        return offsets

    @staticmethod
    def _pairwise_distances(pos: np.ndarray) -> np.ndarray:
        dists = []
        n = pos.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(pos[i] - pos[j]))
        return np.asarray(dists, dtype=np.float32)



