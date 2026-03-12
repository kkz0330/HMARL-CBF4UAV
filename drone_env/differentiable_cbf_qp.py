from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .cbf_qp_matrix import (
    compute_cbf_matrices_centralized,
    compute_clf_tracking_matrices_centralized,
    compute_skill_clf_matrices_centralized,
)


@dataclass
class DifferentiableCBFQPConfig:
    alpha: float = 4.0
    safe_distance: float = 0.22
    slack_weight: float = 50.0
    solve_method: str = "SCS"
    n_jobs_forward: int = 1
    n_jobs_backward: int = 1
    eps: float = 1e-5
    max_iters: int = 5000
    enable_clf: bool = False
    clf_rate: float = 1.0
    clf_deadzone: float = 0.05
    clf_slack_weight: float = 20.0
    clf_mode: str = "position"  # "position" | "skill"
    cruise_speed: float = 0.5
    accelerate_speed: float = 0.9
    decelerate_speed: float = 0.2
    heading_clf_rate: float = 2.0
    speed_clf_deadzone: float = 0.08
    heading_clf_deadzone: float = 0.08


class DifferentiableCBFQPSolver:
    """Differentiable CBF-QP safety filter for multi-UAV velocity commands.

    Interface-compatible with RLCBFQPWrapper qp_solver:
      (cbf_state[N,6], v_des[N,3], last_info) -> (v_safe[N,3], qp_info)
    """

    def __init__(
        self,
        num_drones: int,
        vel_low: np.ndarray,
        vel_high: np.ndarray,
        config: Optional[DifferentiableCBFQPConfig] = None,
        device: str = "cpu",
    ):
        self.num_drones = int(num_drones)
        self.cfg = config or DifferentiableCBFQPConfig()
        self.device = device

        self.vel_low = np.asarray(vel_low, dtype=np.float32).reshape(3)
        self.vel_high = np.asarray(vel_high, dtype=np.float32).reshape(3)
        if np.any(self.vel_low >= self.vel_high):
            raise ValueError("vel_low must be strictly smaller than vel_high per dimension.")

        self.num_pairs = self.num_drones * (self.num_drones - 1) // 2
        self.action_dim = 3 * self.num_drones

        self._torch = None
        self._layer = None
        self._build_layer()

    def _build_layer(self) -> None:
        try:
            import cvxpy as cp
            import torch
            from cvxpylayers.torch import CvxpyLayer
        except Exception as exc:
            raise ImportError(
                "Differentiable CBF-QP requires torch, cvxpy and cvxpylayers."
            ) from exc

        u = cp.Variable(self.action_dim)  # stacked [vx1, vy1, vz1, ..., vxN, vyN, vzN]
        slack_cbf = cp.Variable(self.num_pairs, nonneg=True)
        slack_clf = cp.Variable(self.num_drones, nonneg=True)

        u_nom = cp.Parameter(self.action_dim)
        A_cbf = cp.Parameter((self.num_pairs, self.action_dim))
        b_cbf = cp.Parameter(self.num_pairs)
        A_clf = cp.Parameter((self.num_drones, self.action_dim))
        b_clf = cp.Parameter(self.num_drones)
        u_lb = cp.Parameter(self.action_dim)
        u_ub = cp.Parameter(self.action_dim)

        objective = cp.Minimize(
            cp.sum_squares(u - u_nom)
            + self.cfg.slack_weight * cp.sum_squares(slack_cbf)
            + self.cfg.clf_slack_weight * cp.sum_squares(slack_clf)
        )
        constraints = [
            A_cbf @ u <= b_cbf + slack_cbf,
            A_clf @ u <= b_clf + slack_clf,
            u >= u_lb,
            u <= u_ub,
        ]
        problem = cp.Problem(objective, constraints)

        if not problem.is_dpp():
            raise RuntimeError("CBF-QP problem must satisfy DPP to be differentiable.")

        self._torch = torch
        self._layer = CvxpyLayer(
            problem,
            parameters=[u_nom, A_cbf, b_cbf, A_clf, b_clf, u_lb, u_ub],
            variables=[u, slack_cbf, slack_clf],
        )

    def _build_cbf_Ab(self, cbf_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        A, b, _, d_min, h_min = compute_cbf_matrices_centralized(
            cbf_state,
            safe_distance=self.cfg.safe_distance,
            alpha=self.cfg.alpha,
        )
        return A, b, d_min, h_min

    def _build_clf_Ab(
        self,
        cbf_state: np.ndarray,
        target_pos: np.ndarray | None,
        target_vel: np.ndarray | None,
        skill_idx: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool, float, float]:
        if not self.cfg.enable_clf:
            A = np.zeros((self.num_drones, self.action_dim), dtype=np.float32)
            b = np.full((self.num_drones,), 1e6, dtype=np.float32)
            return A, b, False, np.nan, np.nan

        if self.cfg.clf_mode == "skill":
            if skill_idx is None:
                A = np.zeros((self.num_drones, self.action_dim), dtype=np.float32)
                b = np.full((self.num_drones,), 1e6, dtype=np.float32)
                return A, b, False, np.nan, np.nan
            A, b, mode_flags = compute_skill_clf_matrices_centralized(
                state_n6=cbf_state,
                skill_idx_n=np.asarray(skill_idx, dtype=np.int64).reshape(self.num_drones),
                cruise_speed=self.cfg.cruise_speed,
                accelerate_speed=self.cfg.accelerate_speed,
                decelerate_speed=self.cfg.decelerate_speed,
                speed_clf_rate=self.cfg.clf_rate,
                heading_clf_rate=self.cfg.heading_clf_rate,
                speed_deadzone=self.cfg.speed_clf_deadzone,
                heading_deadzone=self.cfg.heading_clf_deadzone,
            )
            active = mode_flags > 0
            active_ratio = float(np.mean(active.astype(np.float32)))
            heading_ratio = float(np.mean((mode_flags == 2).astype(np.float32)))
            return A, b, True, active_ratio, heading_ratio

        if target_pos is None:
            A = np.zeros((self.num_drones, self.action_dim), dtype=np.float32)
            b = np.full((self.num_drones,), 1e6, dtype=np.float32)
            return A, b, False, np.nan, np.nan

        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(self.num_drones, 3)
        target_vel_n3 = None
        if target_vel is not None:
            target_vel_n3 = np.asarray(target_vel, dtype=np.float32).reshape(self.num_drones, 3)
        A, b, mean_err, max_err = compute_clf_tracking_matrices_centralized(
            state_n6=cbf_state,
            target_n3=target_pos,
            clf_rate=self.cfg.clf_rate,
            target_vel_n3=target_vel_n3,
            deadzone=self.cfg.clf_deadzone,
        )
        return A, b, True, mean_err, max_err

    def solve_torch(self, cbf_state_t, v_des_t, target_pos_t=None, target_vel_t=None, skill_idx_t=None):
        """Differentiable solve (torch tensors)."""
        torch = self._torch
        if torch is None or self._layer is None:
            raise RuntimeError("CvxpyLayer has not been initialized.")

        if v_des_t.ndim != 2 or v_des_t.shape != (self.num_drones, 3):
            raise ValueError(f"v_des_t must have shape {(self.num_drones, 3)}, got {tuple(v_des_t.shape)}")
        if cbf_state_t.ndim != 2 or cbf_state_t.shape[0] != self.num_drones or cbf_state_t.shape[1] < 3:
            raise ValueError(f"cbf_state_t must have shape {(self.num_drones, 6)} (or >=6), got {tuple(cbf_state_t.shape)}")

        cbf_state_np = cbf_state_t.detach().cpu().numpy()
        A_np, b_np, _, _ = self._build_cbf_Ab(cbf_state_np)
        target_pos_np = None if target_pos_t is None else target_pos_t.detach().cpu().numpy()
        target_vel_np = None if target_vel_t is None else target_vel_t.detach().cpu().numpy()
        skill_idx_np = None if skill_idx_t is None else skill_idx_t.detach().cpu().numpy()
        A_clf_np, b_clf_np, _, _, _ = self._build_clf_Ab(cbf_state_np, target_pos_np, target_vel_np, skill_idx_np)
        if not np.isfinite(A_np).all() or not np.isfinite(b_np).all():
            raise RuntimeError("Non-finite CBF matrices detected (A or b).")
        if not np.isfinite(A_clf_np).all() or not np.isfinite(b_clf_np).all():
            raise RuntimeError("Non-finite CLF matrices detected (A_clf or b_clf).")

        # diffcp+SCS is sensitive to dtype; float64 is significantly more stable than float32.
        dtype = torch.float64
        u_nom = v_des_t.reshape(-1).to(dtype=dtype)
        A = torch.as_tensor(np.ascontiguousarray(A_np, dtype=np.float64), dtype=dtype, device=u_nom.device)
        b = torch.as_tensor(np.ascontiguousarray(b_np, dtype=np.float64), dtype=dtype, device=u_nom.device)
        A_clf = torch.as_tensor(np.ascontiguousarray(A_clf_np, dtype=np.float64), dtype=dtype, device=u_nom.device)
        b_clf = torch.as_tensor(np.ascontiguousarray(b_clf_np, dtype=np.float64), dtype=dtype, device=u_nom.device)
        lb = torch.as_tensor(np.tile(self.vel_low, self.num_drones), dtype=dtype, device=u_nom.device)
        ub = torch.as_tensor(np.tile(self.vel_high, self.num_drones), dtype=dtype, device=u_nom.device)

        solver_args = {
            "solve_method": self.cfg.solve_method,
            "eps": self.cfg.eps,
            "max_iters": self.cfg.max_iters,
        }

        u_safe, slack_cbf, slack_clf = self._layer(
            u_nom,
            A,
            b,
            A_clf,
            b_clf,
            lb,
            ub,
            solver_args=solver_args,
        )
        slack_all = torch.cat([slack_cbf, slack_clf], dim=0)
        return u_safe.reshape(self.num_drones, 3).to(v_des_t.dtype), slack_all.to(v_des_t.dtype)

    def __call__(
        self, cbf_state: np.ndarray, v_des: np.ndarray, last_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Wrapper-compatible numpy solve path."""
        v_des = np.asarray(v_des, dtype=np.float32).reshape(self.num_drones, 3)
        cbf_state = np.asarray(cbf_state, dtype=np.float32).reshape(self.num_drones, -1)

        A_np, b_np, d_min, h_min = self._build_cbf_Ab(cbf_state)
        target_pos_np = None
        target_vel_np = None
        if isinstance(last_info, dict):
            target_pos_np = last_info.get("desired_positions", None)
            target_vel_np = last_info.get("target_velocity", None)
            skill_idx_np = last_info.get("skill_idx", None)
        else:
            skill_idx_np = None
        A_clf_np, b_clf_np, clf_on, clf_err_mean, clf_err_max = self._build_clf_Ab(
            cbf_state,
            target_pos_np,
            target_vel_np,
            skill_idx_np,
        )
        torch = self._torch
        if torch is None or self._layer is None:
            return v_des, {"status": "missing_layer", "reason": "cvxpylayers not initialized"}

        try:
            with torch.enable_grad():
                if not np.isfinite(A_np).all() or not np.isfinite(b_np).all():
                    return v_des, {"status": "invalid_cbf_matrix", "reason": "A or b contains non-finite values"}

                dtype = torch.float64
                u_nom = torch.as_tensor(v_des.reshape(-1), dtype=dtype, device=self.device)
                A = torch.as_tensor(np.ascontiguousarray(A_np, dtype=np.float64), dtype=dtype, device=self.device)
                b = torch.as_tensor(np.ascontiguousarray(b_np, dtype=np.float64), dtype=dtype, device=self.device)
                A_clf = torch.as_tensor(np.ascontiguousarray(A_clf_np, dtype=np.float64), dtype=dtype, device=self.device)
                b_clf = torch.as_tensor(np.ascontiguousarray(b_clf_np, dtype=np.float64), dtype=dtype, device=self.device)
                lb = torch.as_tensor(np.tile(self.vel_low, self.num_drones), dtype=dtype, device=self.device)
                ub = torch.as_tensor(np.tile(self.vel_high, self.num_drones), dtype=dtype, device=self.device)

                solver_args = {
                    "solve_method": self.cfg.solve_method,
                    "eps": self.cfg.eps,
                    "max_iters": self.cfg.max_iters,
                }
                u_safe, slack_cbf, slack_clf = self._layer(
                    u_nom,
                    A,
                    b,
                    A_clf,
                    b_clf,
                    lb,
                    ub,
                    solver_args=solver_args,
                )
            v_safe = u_safe.detach().cpu().numpy().reshape(self.num_drones, 3).astype(np.float32)
            slack_cbf_np = slack_cbf.detach().cpu().numpy().astype(np.float32)
            slack_clf_np = slack_clf.detach().cpu().numpy().astype(np.float32)
            slack_np = np.concatenate([slack_cbf_np, slack_clf_np], axis=0)
            info = {
                "status": "ok",
                "min_distance": d_min,
                "min_h": h_min,
                "slack_l2": float(np.linalg.norm(slack_np)),
                "slack_max": float(np.max(slack_np)) if slack_np.size > 0 else 0.0,
                "clf_enabled": bool(clf_on),
                "clf_metric_1": float(clf_err_mean) if np.isfinite(clf_err_mean) else np.nan,
                "clf_metric_2": float(clf_err_max) if np.isfinite(clf_err_max) else np.nan,
                "cbf_constraints": int(self.num_pairs),
                "clf_constraints": int(self.num_drones) if clf_on else 0,
                "clf_mode": self.cfg.clf_mode,
            }
            return v_safe, info
        except Exception as exc:
            return v_des, {"status": "solver_error", "reason": str(exc)}


def build_solver_from_velocity_bounds(
    num_drones: int,
    max_speed_xy: float,
    max_speed_z: float,
    config: Optional[DifferentiableCBFQPConfig] = None,
    device: str = "cpu",
) -> DifferentiableCBFQPSolver:
    vel_low = np.array([-max_speed_xy, -max_speed_xy, -max_speed_z], dtype=np.float32)
    vel_high = np.array([max_speed_xy, max_speed_xy, max_speed_z], dtype=np.float32)
    return DifferentiableCBFQPSolver(
        num_drones=num_drones,
        vel_low=vel_low,
        vel_high=vel_high,
        config=config,
        device=device,
    )
