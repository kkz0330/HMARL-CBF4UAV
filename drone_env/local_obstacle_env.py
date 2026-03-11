from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from gymnasium import spaces

from .formation_env import FormationAviaryEnv, FormationEnvConfig


@dataclass
class ObstacleSpec:
    kind: str
    position: np.ndarray
    radius: float
    half_extents: np.ndarray | None = None
    height: float | None = None
    body_id: int | None = None


@dataclass
class LocalObstacleEnvConfig(FormationEnvConfig):
    sensing_radius: float = 3.0
    max_sensed_obstacles: int = 3
    scenario: str = "bridge_tree"  # options: bridge, tree, bridge_tree, none
    bridge_x: float = 1.8
    bridge_pillar_offset_y: float = 0.40
    bridge_pillar_half_x: float = 0.15
    bridge_pillar_half_y: float = 0.15
    bridge_pillar_half_z: float = 0.8
    bridge_beam_half_z: float = 0.15
    use_bridge_pillar_25d_cbf: bool = True
    bridge_pillar_z_margin: float = 0.08
    use_bridge_beam_25d_cbf: bool = True
    bridge_beam_xy_margin: float = 0.02
    obstacle_clearance: float = 0.25
    obstacle_cbf_alpha: float = 4.0
    include_neighbor_features: bool = True
    _dummy: int = field(default=0, repr=False)  # dataclass inheritance compatibility


class LocalObstacleFormationEnv(FormationAviaryEnv):
    """Formation env with static obstacles and local obstacle observation.

    Observation is per-drone local matrix with shape (N, local_dim):
      [self_pos(3), self_vel(3), desired_err(3), neighbor_feats..., obstacle_feats...]
    """

    def __init__(self, config: LocalObstacleEnvConfig | None = None):
        cfg = config or LocalObstacleEnvConfig()
        super().__init__(cfg)
        self.cfg: LocalObstacleEnvConfig = cfg
        self._obstacles: List[ObstacleSpec] = []

        local_dim = 9
        if self.cfg.include_neighbor_features:
            local_dim += 4 * (self.cfg.num_drones - 1)  # rel_pos xyz + dist
        local_dim += 4 * self.cfg.max_sensed_obstacles  # rel_obs xyz + radius

        obs_low = np.full((self.cfg.num_drones, local_dim), -np.finfo(np.float32).max, dtype=np.float32)
        obs_high = np.full((self.cfg.num_drones, local_dim), np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        # Base reset rebuilds physics world; previous obstacle body ids are invalid.
        # Clear local cache and respawn without trying to remove stale bodies.
        self._obstacles = []
        self._spawn_scene_obstacles(clean_existing=False)

        if self._last_raw_obs is not None:
            obs = self._build_obs(self._last_raw_obs)
            info = self._build_info(
                self._last_raw_obs,
                np.zeros((self.cfg.num_drones, 3), dtype=np.float32),
                np.zeros((self.cfg.num_drones, 3), dtype=np.float32),
            )
        return obs, info

    def close(self):
        self._remove_scene_obstacles()
        super().close()

    def _spawn_scene_obstacles(self, clean_existing: bool = True) -> None:
        if clean_existing:
            self._remove_scene_obstacles()
        if self.cfg.scenario == "none":
            return
        if self._sim_env is None:
            return

        try:
            import pybullet as p
        except ImportError as exc:
            raise ImportError("pybullet is required for obstacle scene spawning.") from exc

        client_id = self._sim_env.CLIENT

        def add_box(center: List[float], half_extents: List[float], rgba: List[float], kind: str):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client_id)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba, physicsClientId=client_id)
            bid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=center,
                physicsClientId=client_id,
            )
            radius = float(np.linalg.norm(np.asarray(half_extents, dtype=np.float32)))
            self._obstacles.append(
                ObstacleSpec(
                    kind=kind,
                    position=np.asarray(center, dtype=np.float32),
                    radius=radius,
                    half_extents=np.asarray(half_extents, dtype=np.float32),
                    body_id=bid,
                )
            )

        def add_cylinder(center: List[float], radius: float, height: float, rgba: List[float], kind: str):
            col = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client_id
            )
            vis = p.createVisualShape(
                p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba, physicsClientId=client_id
            )
            bid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=center,
                physicsClientId=client_id,
            )
            # conservative equivalent sphere radius
            eff_r = float(np.sqrt(radius**2 + (0.5 * height) ** 2))
            self._obstacles.append(
                ObstacleSpec(
                    kind=kind,
                    position=np.asarray(center, dtype=np.float32),
                    radius=eff_r,
                    height=float(height),
                    body_id=bid,
                )
            )

        def add_sphere(center: List[float], radius: float, rgba: List[float], kind: str):
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=client_id)
            bid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=center,
                physicsClientId=client_id,
            )
            self._obstacles.append(
                ObstacleSpec(
                    kind=kind,
                    position=np.asarray(center, dtype=np.float32),
                    radius=float(radius),
                    body_id=bid,
                )
            )

        use_bridge = self.cfg.scenario in {"bridge", "bridge_tree"}
        use_tree = self.cfg.scenario in {"tree", "bridge_tree"}

        if use_bridge:
            # Narrow bridge opening: drones must reconfigure to pass.
            px = float(self.cfg.bridge_x)
            py = float(self.cfg.bridge_pillar_offset_y)
            hx = float(self.cfg.bridge_pillar_half_x)
            hy = float(self.cfg.bridge_pillar_half_y)
            hz = float(self.cfg.bridge_pillar_half_z)
            beam_hz = float(self.cfg.bridge_beam_half_z)
            beam_hy = py + hy

            add_box([px, -py, hz], [hx, hy, hz], [0.6, 0.6, 0.7, 1.0], "bridge_pillar")
            add_box([px, py, hz], [hx, hy, hz], [0.6, 0.6, 0.7, 1.0], "bridge_pillar")
            add_box([px, 0.0, 2.0 * hz + beam_hz], [hx, beam_hy, beam_hz], [0.5, 0.5, 0.6, 1.0], "bridge_beam")

        if use_tree:
            # Trunk + canopy.
            add_cylinder([3.4, 0.0, 0.6], radius=0.18, height=1.2, rgba=[0.45, 0.28, 0.12, 1.0], kind="tree_trunk")
            add_sphere([3.4, 0.0, 1.55], radius=0.55, rgba=[0.1, 0.6, 0.15, 1.0], kind="tree_canopy")

    def _remove_scene_obstacles(self) -> None:
        if self._sim_env is None:
            self._obstacles = []
            return
        if not self._obstacles:
            return

        try:
            import pybullet as p
        except ImportError:
            self._obstacles = []
            return

        client_id = self._sim_env.CLIENT
        for obs in self._obstacles:
            if obs.body_id is not None:
                try:
                    # Avoid pybullet warning spam on already-invalid ids.
                    p.getBodyInfo(obs.body_id, physicsClientId=client_id)
                    p.removeBody(obs.body_id, physicsClientId=client_id)
                except Exception:
                    pass
        self._obstacles = []

    def _build_obs(self, raw_obs: Any) -> np.ndarray:
        pos, vel = self._parse_pos_vel(raw_obs)
        desired = self._current_desired_positions()
        err = desired - pos

        local_obs = []
        for i in range(self.cfg.num_drones):
            feat: List[float] = []
            feat.extend(pos[i].tolist())
            feat.extend(vel[i].tolist())
            feat.extend(err[i].tolist())

            if self.cfg.include_neighbor_features:
                for j in range(self.cfg.num_drones):
                    if j == i:
                        continue
                    rel = pos[j] - pos[i]
                    feat.extend(rel.tolist())
                    feat.append(float(np.linalg.norm(rel)))

            sensed = self.get_sensed_obstacles_for_drone(pos[i])
            for k in range(self.cfg.max_sensed_obstacles):
                if k < len(sensed):
                    obs = sensed[k]
                    rel = obs.position - pos[i]
                    feat.extend(rel.tolist())
                    feat.append(float(obs.radius))
                else:
                    feat.extend([0.0, 0.0, 0.0, 0.0])

            local_obs.append(feat)
        return np.asarray(local_obs, dtype=np.float32)

    def _build_info(self, raw_obs: Any, action: np.ndarray, target_vel: np.ndarray) -> Dict[str, Any]:
        info = super()._build_info(raw_obs, action, target_vel)
        pos, _ = self._parse_pos_vel(raw_obs)

        sensed_counts = []
        min_clearance = float("inf")
        sensed_payload = []
        for i in range(self.cfg.num_drones):
            sensed = self.get_sensed_obstacles_for_drone(pos[i])
            sensed_counts.append(len(sensed))
            drone_payload = []
            for obs in sensed:
                center_dist = float(np.linalg.norm(obs.position - pos[i]))
                clearance = self._compute_obstacle_clearance(pos[i], obs)
                min_clearance = min(min_clearance, clearance)
                drone_payload.append(
                    {
                        "kind": obs.kind,
                        "position": obs.position.copy(),
                        "radius": float(obs.radius),
                        "center_distance": center_dist,
                        "clearance": clearance,
                    }
                )
            sensed_payload.append(drone_payload)

        if min_clearance == float("inf"):
            min_clearance = float("inf")

        contact_count = self._count_obstacle_contacts()
        A_obs, b_obs = self.get_obstacle_cbf_matrices(raw_obs=raw_obs)
        info.update(
            {
                "sensing_radius": float(self.cfg.sensing_radius),
                "max_sensed_obstacles": int(self.cfg.max_sensed_obstacles),
                "sensed_obstacle_count": np.asarray(sensed_counts, dtype=np.int32),
                "sensed_obstacles": sensed_payload,
                "min_obstacle_clearance": float(min_clearance),
                "obstacle_cbf_A": A_obs,
                "obstacle_cbf_b": b_obs,
                "obstacle_contact_count": int(contact_count),
                "obstacles_total": len(self._obstacles),
            }
        )
        return info

    def _count_obstacle_contacts(self) -> int:
        if self._sim_env is None or not self._obstacles:
            return 0
        try:
            import pybullet as p
        except ImportError:
            return 0

        drone_ids = getattr(self._sim_env, "DRONE_IDS", None)
        if drone_ids is None:
            return 0

        count = 0
        client_id = self._sim_env.CLIENT
        for did in drone_ids:
            for obs in self._obstacles:
                if obs.body_id is None:
                    continue
                try:
                    contacts = p.getContactPoints(bodyA=did, bodyB=obs.body_id, physicsClientId=client_id)
                except Exception:
                    contacts = []
                if contacts:
                    count += len(contacts)
        return count

    def get_sensed_obstacles_for_drone(self, drone_pos: np.ndarray) -> List[ObstacleSpec]:
        if not self._obstacles:
            return []
        sensed = []
        for obs in self._obstacles:
            sense_dist = self._distance_for_sensing(drone_pos, obs)
            if sense_dist <= self.cfg.sensing_radius + self._obstacle_xy_radius(obs):
                sensed.append((sense_dist, obs))
        sensed.sort(key=lambda x: x[0])
        return [x[1] for x in sensed[: self.cfg.max_sensed_obstacles]]

    def _obstacle_xy_radius(self, obs: ObstacleSpec) -> float:
        if obs.kind in {"bridge_pillar", "bridge_beam"} and obs.half_extents is not None:
            return float(np.linalg.norm(obs.half_extents[0:2]))
        return float(obs.radius)

    def _distance_for_sensing(self, drone_pos: np.ndarray, obs: ObstacleSpec) -> float:
        if obs.kind == "bridge_pillar" and obs.half_extents is not None:
            return float(np.linalg.norm(drone_pos[0:2] - obs.position[0:2]))
        return float(np.linalg.norm(drone_pos - obs.position))

    def _bridge_pillar_height_band(self, obs: ObstacleSpec) -> Tuple[float, float]:
        if obs.half_extents is None:
            return -np.inf, np.inf
        z_half = float(obs.half_extents[2])
        z_margin = float(self.cfg.bridge_pillar_z_margin)
        z_low = float(obs.position[2] - z_half - z_margin)
        z_high = float(obs.position[2] + z_half + z_margin)
        return z_low, z_high

    def _compute_obstacle_clearance(self, drone_pos: np.ndarray, obs: ObstacleSpec) -> float:
        if self.cfg.use_bridge_pillar_25d_cbf and obs.kind == "bridge_pillar" and obs.half_extents is not None:
            z_low, z_high = self._bridge_pillar_height_band(obs)
            if drone_pos[2] < z_low or drone_pos[2] > z_high:
                return float("inf")
            dp_xy = drone_pos[0:2] - obs.position[0:2]
            r_eff_xy = self._obstacle_xy_radius(obs) + float(self.cfg.obstacle_clearance)
            return float(np.linalg.norm(dp_xy) - r_eff_xy)

        if self.cfg.use_bridge_beam_25d_cbf and obs.kind == "bridge_beam" and obs.half_extents is not None:
            hx = float(obs.half_extents[0] + self.cfg.bridge_beam_xy_margin)
            hy = float(obs.half_extents[1] + self.cfg.bridge_beam_xy_margin)
            if abs(float(drone_pos[0] - obs.position[0])) > hx or abs(float(drone_pos[1] - obs.position[1])) > hy:
                return float("inf")
            z_limit = float(obs.position[2] - obs.half_extents[2] - self.cfg.obstacle_clearance)
            return float(z_limit - drone_pos[2])

        center_dist = float(np.linalg.norm(obs.position - drone_pos))
        return float(center_dist - float(obs.radius) - float(self.cfg.obstacle_clearance))

    def _build_single_obstacle_cbf(self, drone_pos: np.ndarray, obs: ObstacleSpec) -> Tuple[np.ndarray, float] | None:
        if self.cfg.use_bridge_pillar_25d_cbf and obs.kind == "bridge_pillar" and obs.half_extents is not None:
            z_low, z_high = self._bridge_pillar_height_band(obs)
            if drone_pos[2] < z_low or drone_pos[2] > z_high:
                return None
            dp_xy = drone_pos[0:2] - obs.position[0:2]
            r_eff_xy = self._obstacle_xy_radius(obs) + float(self.cfg.obstacle_clearance)
            h_io = float(np.dot(dp_xy, dp_xy) - r_eff_xy**2)
            A = np.array([-2.0 * dp_xy[0], -2.0 * dp_xy[1], 0.0], dtype=np.float32)
            b = float(self.cfg.obstacle_cbf_alpha * h_io)
            return A, b

        if self.cfg.use_bridge_beam_25d_cbf and obs.kind == "bridge_beam" and obs.half_extents is not None:
            hx = float(obs.half_extents[0] + self.cfg.bridge_beam_xy_margin)
            hy = float(obs.half_extents[1] + self.cfg.bridge_beam_xy_margin)
            if abs(float(drone_pos[0] - obs.position[0])) > hx or abs(float(drone_pos[1] - obs.position[1])) > hy:
                return None
            z_limit = float(obs.position[2] - obs.half_extents[2] - self.cfg.obstacle_clearance)
            h_io = float(z_limit - drone_pos[2])
            # Keep altitude below the beam underside: u_z <= alpha * (z_limit - z)
            A = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            b = float(self.cfg.obstacle_cbf_alpha * h_io)
            return A, b

        dp = drone_pos - obs.position
        r_eff = float(obs.radius + self.cfg.obstacle_clearance)
        h_io = float(np.dot(dp, dp) - r_eff**2)
        A = (-2.0 * dp).astype(np.float32)
        b = float(self.cfg.obstacle_cbf_alpha * h_io)
        return A, b

    def get_obstacle_cbf_matrices(
        self, raw_obs: Any | None = None, cbf_state: np.ndarray | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Distributed obstacle CBF constraints per drone.

        Returns:
          A_obs: (N, K, 3)
          b_obs: (N, K)
        Constraint:
          A_obs[i,k] @ u_i <= b_obs[i,k]
          where h = ||p_i - p_o||^2 - (r_o + clearance)^2
        """
        if cbf_state is None:
            if raw_obs is None:
                if self._last_raw_obs is None:
                    raise RuntimeError("No state available to build obstacle CBF matrices.")
                raw_obs = self._last_raw_obs
            cbf_state = self.get_cbf_state(raw_obs)
        pos = np.asarray(cbf_state, dtype=np.float32)[:, 0:3]

        n = self.cfg.num_drones
        kmax = self.cfg.max_sensed_obstacles
        A_obs = np.zeros((n, kmax, 3), dtype=np.float32)
        b_obs = np.zeros((n, kmax), dtype=np.float32)

        for i in range(n):
            sensed = self.get_sensed_obstacles_for_drone(pos[i])
            k = 0
            for obs in sensed:
                if k >= kmax:
                    break
                con = self._build_single_obstacle_cbf(pos[i], obs)
                if con is None:
                    continue
                A_k, b_k = con
                A_obs[i, k, :] = A_k
                b_obs[i, k] = b_k
                k += 1
        return A_obs, b_obs
