from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compute_cbf_matrices_distributed(
    state_n6: np.ndarray, safe_distance: float = 0.22, alpha: float = 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-agent CBF linear constraints for distributed QP.

    Returns:
      A_batch: shape (N, N-1, 3)
      b_batch: shape (N, N-1)
    Constraint for agent i against neighbor j:
      A_ij u_i <= b_ij
      A_ij = -2 (p_i - p_j)^T
      b_ij = alpha*h_ij - 2 (p_i - p_j)^T v_j
      h_ij = ||p_i - p_j||^2 - d_safe^2
    """
    state = np.asarray(state_n6, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] < 6:
        raise ValueError(f"state_n6 must have shape (N,6+), got {state.shape}")

    n = state.shape[0]
    pos = state[:, 0:3]
    vel = state[:, 3:6]

    A_batch = np.zeros((n, max(n - 1, 0), 3), dtype=np.float32)
    b_batch = np.zeros((n, max(n - 1, 0)), dtype=np.float32)

    for i in range(n):
        k = 0
        for j in range(n):
            if i == j:
                continue
            dp = pos[i] - pos[j]
            h_ij = float(np.dot(dp, dp) - safe_distance**2)
            A_batch[i, k, :] = -2.0 * dp
            b_batch[i, k] = alpha * h_ij - 2.0 * float(np.dot(dp, vel[j]))
            k += 1
    return A_batch, b_batch


def compute_cbf_matrices_centralized(
    state_n6: np.ndarray, safe_distance: float = 0.22, alpha: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], float, float]:
    """Build pairwise CBF constraints for centralized multi-agent QP.

    Decision vector is stacked velocities:
      u = [u_1, ..., u_N] in R^(3N)
    For each pair (i,j), constraint is:
      -2*(p_i-p_j)^T (u_i - u_j) <= alpha * h_ij

    Returns:
      A: shape (N_pairs, 3N)
      b: shape (N_pairs,)
      pair_indices: list of (i,j)
      min_distance: minimum pairwise Euclidean distance
      min_h: minimum CBF value h_ij
    """
    state = np.asarray(state_n6, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] < 6:
        raise ValueError(f"state_n6 must have shape (N,6+), got {state.shape}")

    n = state.shape[0]
    pos = state[:, 0:3]
    pair_indices: List[Tuple[int, int]] = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(pair_indices)

    A = np.zeros((m, 3 * n), dtype=np.float32)
    b = np.zeros((m,), dtype=np.float32)
    min_distance = np.inf
    min_h = np.inf

    for k, (i, j) in enumerate(pair_indices):
        dp = pos[i] - pos[j]
        d_ij = float(np.linalg.norm(dp))
        h_ij = float(np.dot(dp, dp) - safe_distance**2)

        A[k, 3 * i : 3 * i + 3] = -2.0 * dp
        A[k, 3 * j : 3 * j + 3] = 2.0 * dp
        b[k] = alpha * h_ij

        min_distance = min(min_distance, d_ij)
        min_h = min(min_h, h_ij)

    if m == 0:
        min_distance = np.inf
        min_h = np.inf

    return A, b, pair_indices, float(min_distance), float(min_h)


def compute_clf_tracking_matrices_centralized(
    state_n6: np.ndarray,
    target_n3: np.ndarray,
    clf_rate: float = 1.0,
    target_vel_n3: np.ndarray | None = None,
    deadzone: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Build centralized per-agent CLF tracking constraints.

    Decision vector is stacked velocities:
      u = [u_1, ..., u_N] in R^(3N)

    For agent i, define tracking error:
      e_i = p_i - p_i^*
      V_i = 0.5 * ||e_i||^2
      e_dot_i = u_i - v_i^*

    CLF inequality:
      dV_i/dt <= -clf_rate * V_i
      e_i^T u_i <= e_i^T v_i^* - 0.5 * clf_rate * ||e_i||^2

    Returns:
      A_clf: shape (N, 3N)
      b_clf: shape (N,)
      mean_error_norm: mean ||e_i||
      max_error_norm: max ||e_i||
    """
    state = np.asarray(state_n6, dtype=np.float32)
    target = np.asarray(target_n3, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] < 3:
        raise ValueError(f"state_n6 must have shape (N,3+), got {state.shape}")
    if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] != state.shape[0]:
        raise ValueError(f"target_n3 must have shape {(state.shape[0], 3)}, got {target.shape}")
    if clf_rate < 0.0:
        raise ValueError("clf_rate must be non-negative.")

    n = state.shape[0]
    pos = state[:, 0:3]
    err = pos - target

    if target_vel_n3 is None:
        target_vel = np.zeros((n, 3), dtype=np.float32)
    else:
        target_vel = np.asarray(target_vel_n3, dtype=np.float32)
        if target_vel.shape != (n, 3):
            raise ValueError(f"target_vel_n3 must have shape {(n, 3)}, got {target_vel.shape}")

    A = np.zeros((n, 3 * n), dtype=np.float32)
    b = np.full((n,), 1e6, dtype=np.float32)  # relaxed by default (used for deadzone rows)

    err_norm = np.linalg.norm(err, axis=1)
    dz = max(float(deadzone), 0.0)
    for i in range(n):
        if err_norm[i] <= dz:
            continue
        e_i = err[i]
        A[i, 3 * i : 3 * i + 3] = e_i
        b[i] = float(np.dot(e_i, target_vel[i]) - 0.5 * clf_rate * np.dot(e_i, e_i))

    return A, b, float(np.mean(err_norm)), float(np.max(err_norm))


def compute_skill_clf_matrices_centralized(
    state_n6: np.ndarray,
    skill_idx_n: np.ndarray,
    cruise_speed: float = 0.5,
    accelerate_speed: float = 0.2,
    decelerate_speed: float = 0.2,
    skill_speed_cap: float = 1.0,
    speed_clf_rate: float = 1.0,
    heading_clf_rate: float = 2.0,
    speed_deadzone: float = 0.08,
    heading_deadzone: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-agent skill-conditioned CLF constraints.

    Skill mapping:
      0 cruise, 1 left, 2 right, 3 up, 4 down, 5 accelerate, 6 decelerate

    Speed-skill semantics:
      accelerate/decelerate use speed increments based on current forward speed v_x.
      v_ref_x = clip(v_x +/- delta, 0, skill_speed_cap)

    Returns:
      A_clf: (N, 3N), b_clf: (N,), mode_flags: (N,)
      mode_flags: 0 relaxed, 1 speed-clf, 2 heading-clf
    """
    state = np.asarray(state_n6, dtype=np.float32)
    skills = np.asarray(skill_idx_n, dtype=np.int64).reshape(-1)
    if state.ndim != 2 or state.shape[1] < 6:
        raise ValueError(f"state_n6 must have shape (N,6+), got {state.shape}")
    n = state.shape[0]
    if skills.shape[0] != n:
        raise ValueError(f"skill_idx_n must have length {n}, got {skills.shape}")

    vel = state[:, 3:6]
    A = np.zeros((n, 3 * n), dtype=np.float32)
    b = np.full((n,), 1e6, dtype=np.float32)
    mode_flags = np.zeros((n,), dtype=np.int32)

    d_forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    d_left = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    d_right = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    d_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    d_down = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    speed_deadzone = max(float(speed_deadzone), 0.0)
    heading_deadzone = max(float(heading_deadzone), 0.0)
    speed_cap = max(float(skill_speed_cap), 0.0)
    accel_delta = max(float(accelerate_speed), 0.0)
    decel_delta = max(float(decelerate_speed), 0.0)

    for i in range(n):
        z = int(skills[i])
        v_i = vel[i]

        # Speed-type CLF (cruise / accelerate / decelerate)
        if z in (0, 5, 6):
            if z == 0:
                speed_ref = float(cruise_speed)
            elif z == 5:
                speed_ref = float(np.clip(max(float(v_i[0]), 0.0) + accel_delta, 0.0, speed_cap))
            else:
                speed_ref = float(np.clip(max(float(v_i[0]), 0.0) - decel_delta, 0.0, speed_cap))
            v_ref = speed_ref * d_forward
            e = v_i - v_ref
            e_norm = float(np.linalg.norm(e))
            if e_norm <= speed_deadzone:
                continue
            # e^T(u - v) <= -rate*0.5||e||^2  ->  e^T u <= e^T v - rate*0.5||e||^2
            A[i, 3 * i : 3 * i + 3] = e
            b[i] = float(np.dot(e, v_i) - 0.5 * float(speed_clf_rate) * np.dot(e, e))
            mode_flags[i] = 1
            continue

        # Heading-type CLF (left / right / up / down)
        if z == 1:
            d = d_left
        elif z == 2:
            d = d_right
        elif z == 3:
            d = d_up
        elif z == 4:
            d = d_down
        else:
            continue

        # V = 0.5 ||P_d v||^2, where P_d = I - d d^T
        # dV/dt = (P_d v)^T P_d (u - v) <= -rate*V
        # Since P_d is symmetric idempotent:
        # (P_d v)^T u <= (1 - 0.5*rate) ||P_d v||^2
        P = np.eye(3, dtype=np.float32) - np.outer(d, d)
        e_dir = P @ v_i
        e_dir_norm = float(np.linalg.norm(e_dir))
        if e_dir_norm <= heading_deadzone:
            continue
        A[i, 3 * i : 3 * i + 3] = e_dir
        b[i] = float((1.0 - 0.5 * float(heading_clf_rate)) * np.dot(e_dir, e_dir))
        mode_flags[i] = 2

    return A, b, mode_flags
