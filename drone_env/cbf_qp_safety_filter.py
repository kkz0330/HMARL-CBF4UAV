from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .cbf_qp_matrix import (
    compute_cbf_matrices_centralized,
    compute_clf_tracking_matrices_centralized,
)


@dataclass
class CBFQPSafetyFilterConfig:
    alpha: float = 4.0
    safe_distance: float = 0.22
    slack_weight: float = 80.0  # deprecated, kept for backward compatibility
    solver_primary: str = "OSQP"
    solver_fallback: str = "SCS"
    enforce_obstacle_constraints: bool = True
    enable_clf: bool = False
    clf_rate: float = 1.0
    clf_deadzone: float = 0.05
    clf_slack_weight: float = 20.0
    clf_slack_l2_weight: float = 1e-4


class CBFQPSafetyFilter:
    """Numpy/CVXPY CBF-QP safety filter.

    Wrapper-compatible signature:
      (cbf_state[N,6], v_des[N,3], last_info) -> (v_safe[N,3], qp_info)
    """

    def __init__(
        self,
        num_drones: int,
        vel_low: np.ndarray,
        vel_high: np.ndarray,
        config: CBFQPSafetyFilterConfig | None = None,
    ):
        self.num_drones = int(num_drones)
        self.cfg = config or CBFQPSafetyFilterConfig()
        self.vel_low = np.asarray(vel_low, dtype=np.float32).reshape(3)
        self.vel_high = np.asarray(vel_high, dtype=np.float32).reshape(3)
        if np.any(self.vel_low >= self.vel_high):
            raise ValueError("vel_low must be strictly smaller than vel_high per dimension.")

    def _obstacle_constraints_from_info(self, info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.cfg.enforce_obstacle_constraints:
            return np.zeros((0, 3 * self.num_drones), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        A_obs = info.get("obstacle_cbf_A", None)
        b_obs = info.get("obstacle_cbf_b", None)
        if A_obs is None or b_obs is None:
            return np.zeros((0, 3 * self.num_drones), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        A_obs = np.asarray(A_obs, dtype=np.float32)
        b_obs = np.asarray(b_obs, dtype=np.float32)
        if A_obs.ndim != 3 or b_obs.ndim != 2:
            return np.zeros((0, 3 * self.num_drones), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        if A_obs.shape[0] != self.num_drones or b_obs.shape[0] != self.num_drones:
            return np.zeros((0, 3 * self.num_drones), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        if A_obs.shape[1] != b_obs.shape[1] or A_obs.shape[2] != 3:
            return np.zeros((0, 3 * self.num_drones), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        kmax = A_obs.shape[1]
        rows = self.num_drones * kmax
        A = np.zeros((rows, 3 * self.num_drones), dtype=np.float32)
        b = b_obs.reshape(-1).astype(np.float32)
        for i in range(self.num_drones):
            A[i * kmax : (i + 1) * kmax, 3 * i : 3 * i + 3] = A_obs[i]
        return A, b

    def _clf_constraints_from_info(self, cbf_state: np.ndarray, info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, bool]:
        if not self.cfg.enable_clf:
            return np.zeros((0, 3 * self.num_drones), dtype=np.float32), np.zeros((0,), dtype=np.float32), False
        target_pos = info.get("desired_positions", None)
        if target_pos is None:
            return np.zeros((0, 3 * self.num_drones), dtype=np.float32), np.zeros((0,), dtype=np.float32), False
        target_vel = info.get("target_velocity", None)
        A_clf, b_clf, _, _ = compute_clf_tracking_matrices_centralized(
            state_n6=cbf_state,
            target_n3=np.asarray(target_pos, dtype=np.float32).reshape(self.num_drones, 3),
            clf_rate=self.cfg.clf_rate,
            target_vel_n3=None if target_vel is None else np.asarray(target_vel, dtype=np.float32).reshape(self.num_drones, 3),
            deadzone=self.cfg.clf_deadzone,
        )
        return A_clf.astype(np.float32), b_clf.astype(np.float32), True

    def _fallback_velocity(self, like_v: np.ndarray) -> np.ndarray:
        v = np.zeros_like(like_v, dtype=np.float32)
        return np.clip(v, self.vel_low[None, :], self.vel_high[None, :]).astype(np.float32)

    def __call__(
        self, cbf_state: np.ndarray, v_des: np.ndarray, last_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        v_des = np.asarray(v_des, dtype=np.float32).reshape(self.num_drones, 3)
        cbf_state = np.asarray(cbf_state, dtype=np.float32).reshape(self.num_drones, -1)

        A_pair, b_pair, _, min_d, min_h = compute_cbf_matrices_centralized(
            cbf_state,
            safe_distance=self.cfg.safe_distance,
            alpha=self.cfg.alpha,
        )
        A_obs, b_obs = self._obstacle_constraints_from_info(last_info)
        A_clf, b_clf, clf_on = self._clf_constraints_from_info(cbf_state, last_info)

        n_u = 3 * self.num_drones
        n_c_hard = int(A_pair.shape[0] + A_obs.shape[0])
        n_c_clf = int(A_clf.shape[0]) if clf_on else 0
        if n_c_hard == 0 and n_c_clf == 0:
            clipped = np.clip(v_des, self.vel_low[None, :], self.vel_high[None, :]).astype(np.float32)
            return clipped, {"status": "ok_no_constraints", "min_distance": float(min_d), "min_h": float(min_h)}

        try:
            import cvxpy as cp
        except Exception as exc:
            return self._fallback_velocity(v_des), {"status": "missing_cvxpy", "reason": str(exc)}

        u = cp.Variable(n_u)
        u_nom = v_des.reshape(-1)
        lb = np.tile(self.vel_low, self.num_drones)
        ub = np.tile(self.vel_high, self.num_drones)

        obj_expr = cp.sum_squares(u - u_nom)
        cons = [u >= lb, u <= ub]
        if A_pair.shape[0] > 0:
            cons.append(A_pair @ u <= b_pair)
        if A_obs.shape[0] > 0:
            cons.append(A_obs @ u <= b_obs)

        slack_clf = None
        if clf_on and A_clf.shape[0] > 0:
            slack_clf = cp.Variable(A_clf.shape[0], nonneg=True)
            cons.append(A_clf @ u <= b_clf + slack_clf)
            obj_expr = obj_expr + float(self.cfg.clf_slack_weight) * cp.sum(slack_clf)
            if float(self.cfg.clf_slack_l2_weight) > 0.0:
                obj_expr = obj_expr + float(self.cfg.clf_slack_l2_weight) * cp.sum_squares(slack_clf)

        obj = cp.Minimize(obj_expr)
        prob = cp.Problem(obj, cons)

        try:
            prob.solve(solver=getattr(cp, self.cfg.solver_primary), warm_start=True, verbose=False)
            if u.value is None:
                prob.solve(solver=getattr(cp, self.cfg.solver_fallback), warm_start=True, verbose=False)
        except Exception as exc:
            return self._fallback_velocity(v_des), {"status": "solver_error", "reason": str(exc)}

        if u.value is None:
            return self._fallback_velocity(v_des), {
                "status": "infeasible",
                "min_distance": float(min_d),
                "min_h": float(min_h),
                "constraints_pair": int(A_pair.shape[0]),
                "constraints_obs": int(A_obs.shape[0]),
                "constraints_clf": int(A_clf.shape[0]) if clf_on else 0,
            }

        if slack_clf is None:
            slack_v = np.zeros((0,), dtype=np.float32)
        else:
            slack_v = np.asarray(slack_clf.value if slack_clf.value is not None else np.zeros(n_c_clf), dtype=np.float32)
        v_safe = np.asarray(u.value, dtype=np.float32).reshape(self.num_drones, 3)
        return v_safe, {
            "status": "ok",
            "min_distance": float(min_d),
            "min_h": float(min_h),
            "constraints_pair": int(A_pair.shape[0]),
            "constraints_obs": int(A_obs.shape[0]),
            "constraints_clf": int(A_clf.shape[0]) if clf_on else 0,
            "clf_enabled": bool(clf_on),
            "slack_l2": float(np.linalg.norm(slack_v)),
            "slack_max": float(np.max(slack_v)) if slack_v.size > 0 else 0.0,
        }
