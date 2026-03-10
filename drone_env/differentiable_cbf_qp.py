from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .cbf_qp_matrix import compute_cbf_matrices_centralized


@dataclass
class DifferentiableCBFQPConfig:
    alpha: float = 4.0
    safe_distance: float = 0.22
    slack_weight: float = 50.0
    solve_method: str = "SCS"
    n_jobs_forward: int = 1
    n_jobs_backward: int = 1
    eps: float = 1e-5


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
        slack = cp.Variable(self.num_pairs, nonneg=True)

        u_nom = cp.Parameter(self.action_dim)
        A_cbf = cp.Parameter((self.num_pairs, self.action_dim))
        b_cbf = cp.Parameter(self.num_pairs)
        u_lb = cp.Parameter(self.action_dim)
        u_ub = cp.Parameter(self.action_dim)

        objective = cp.Minimize(cp.sum_squares(u - u_nom) + self.cfg.slack_weight * cp.sum_squares(slack))
        constraints = [
            A_cbf @ u <= b_cbf + slack,
            u >= u_lb,
            u <= u_ub,
        ]
        problem = cp.Problem(objective, constraints)

        if not problem.is_dpp():
            raise RuntimeError("CBF-QP problem must satisfy DPP to be differentiable.")

        self._torch = torch
        self._layer = CvxpyLayer(
            problem,
            parameters=[u_nom, A_cbf, b_cbf, u_lb, u_ub],
            variables=[u, slack],
        )

    def _build_cbf_Ab(self, cbf_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        A, b, _, d_min, h_min = compute_cbf_matrices_centralized(
            cbf_state,
            safe_distance=self.cfg.safe_distance,
            alpha=self.cfg.alpha,
        )
        return A, b, d_min, h_min

    def solve_torch(self, cbf_state_t, v_des_t):
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

        u_nom = v_des_t.reshape(-1)
        A = torch.as_tensor(A_np, dtype=u_nom.dtype, device=u_nom.device)
        b = torch.as_tensor(b_np, dtype=u_nom.dtype, device=u_nom.device)
        lb = torch.as_tensor(np.tile(self.vel_low, self.num_drones), dtype=u_nom.dtype, device=u_nom.device)
        ub = torch.as_tensor(np.tile(self.vel_high, self.num_drones), dtype=u_nom.dtype, device=u_nom.device)

        u_safe, slack = self._layer(
            u_nom,
            A,
            b,
            lb,
            ub,
            solver_args={
                "solve_method": self.cfg.solve_method,
                "n_jobs_forward": self.cfg.n_jobs_forward,
                "n_jobs_backward": self.cfg.n_jobs_backward,
                "eps": self.cfg.eps,
            },
        )
        return u_safe.reshape(self.num_drones, 3), slack

    def __call__(
        self, cbf_state: np.ndarray, v_des: np.ndarray, last_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Wrapper-compatible numpy solve path."""
        v_des = np.asarray(v_des, dtype=np.float32).reshape(self.num_drones, 3)
        cbf_state = np.asarray(cbf_state, dtype=np.float32).reshape(self.num_drones, -1)

        A_np, b_np, d_min, h_min = self._build_cbf_Ab(cbf_state)
        torch = self._torch
        if torch is None or self._layer is None:
            return v_des, {"status": "missing_layer", "reason": "cvxpylayers not initialized"}

        try:
            with torch.enable_grad():
                u_nom = torch.as_tensor(v_des.reshape(-1), dtype=torch.float32, device=self.device)
                A = torch.as_tensor(A_np, dtype=torch.float32, device=self.device)
                b = torch.as_tensor(b_np, dtype=torch.float32, device=self.device)
                lb = torch.as_tensor(np.tile(self.vel_low, self.num_drones), dtype=torch.float32, device=self.device)
                ub = torch.as_tensor(np.tile(self.vel_high, self.num_drones), dtype=torch.float32, device=self.device)

                u_safe, slack = self._layer(
                    u_nom,
                    A,
                    b,
                    lb,
                    ub,
                    solver_args={
                        "solve_method": self.cfg.solve_method,
                        "n_jobs_forward": self.cfg.n_jobs_forward,
                        "n_jobs_backward": self.cfg.n_jobs_backward,
                        "eps": self.cfg.eps,
                    },
                )
            v_safe = u_safe.detach().cpu().numpy().reshape(self.num_drones, 3).astype(np.float32)
            slack_np = slack.detach().cpu().numpy().astype(np.float32)
            info = {
                "status": "ok",
                "min_distance": d_min,
                "min_h": h_min,
                "slack_l2": float(np.linalg.norm(slack_np)),
                "slack_max": float(np.max(slack_np)) if slack_np.size > 0 else 0.0,
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
