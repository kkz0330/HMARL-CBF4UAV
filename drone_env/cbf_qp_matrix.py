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
