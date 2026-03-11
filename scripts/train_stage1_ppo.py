import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drone_env import (  # noqa: E402
    DifferentiableCBFQPConfig,
    FormationAviaryEnv,
    FormationEnvConfig,
    build_solver_from_velocity_bounds,
)


@dataclass
class PPOConfig:
    rollout_steps: int = 512
    epochs: int = 8
    minibatch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.8
    qp_aux_weight: float = 0.05


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value


class NumpyCBFQPSolver:
    """Non-differentiable but stable CBF-QP solver (recommended on Windows)."""

    def __init__(self, num_drones: int, safe_distance: float, alpha: float, vel_low: np.ndarray, vel_high: np.ndarray):
        self.num_drones = num_drones
        self.safe_distance = safe_distance
        self.alpha = alpha
        self.vel_low = np.asarray(vel_low, dtype=np.float32).reshape(3)
        self.vel_high = np.asarray(vel_high, dtype=np.float32).reshape(3)

    def __call__(self, cbf_state: np.ndarray, v_des: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        from drone_env import compute_cbf_matrices_centralized

        v_des = np.asarray(v_des, dtype=np.float32).reshape(self.num_drones, 3)
        A, b, _, min_d, min_h = compute_cbf_matrices_centralized(
            cbf_state, safe_distance=self.safe_distance, alpha=self.alpha
        )
        n_u = 3 * self.num_drones
        n_c = A.shape[0]

        u = cp.Variable(n_u)
        slack = cp.Variable(n_c, nonneg=True)
        u_nom = v_des.reshape(-1)
        lb = np.tile(self.vel_low, self.num_drones)
        ub = np.tile(self.vel_high, self.num_drones)
        obj = cp.Minimize(cp.sum_squares(u - u_nom) + 50.0 * cp.sum_squares(slack))
        cons = [A @ u <= b + slack, u >= lb, u <= ub]
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if u.value is None:
            prob.solve(solver=cp.SCS, warm_start=True, verbose=False)
        if u.value is None:
            return v_des, {"status": "infeasible", "min_distance": float(min_d), "min_h": float(min_h), "slack_l2": np.nan}
        slack_v = np.asarray(slack.value if slack.value is not None else np.zeros(n_c), dtype=np.float32)
        return (
            np.asarray(u.value, dtype=np.float32).reshape(self.num_drones, 3),
            {"status": "ok", "min_distance": float(min_d), "min_h": float(min_h), "slack_l2": float(np.linalg.norm(slack_v))},
        )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def batch_indices(n: int, batch_size: int):
    perm = np.random.permutation(n)
    for s in range(0, n, batch_size):
        yield perm[s : s + batch_size]


def main():
    parser = argparse.ArgumentParser(description="Stage-1 flat PPO with CBF-QP safety layer")
    parser.add_argument("--num-drones", type=int, default=3)
    parser.add_argument("--total-updates", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--diff-qp",
        action="store_true",
        default=False,
        help="Use cvxpylayers/diffcp differentiable QP (recommended in WSL/Linux, unstable on Windows).",
    )
    args = parser.parse_args()

    if os.name == "nt" and args.diff_qp:
        raise SystemExit(
            "Windows detected: --diff-qp may crash due to diffcp/cvxpylayers native issue.\n"
            "Run without --diff-qp on Windows, or run this script inside WSL/Linux Python."
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ppo_cfg = PPOConfig(rollout_steps=args.rollout_steps)
    device = torch.device(args.device)

    env = FormationAviaryEnv(
        FormationEnvConfig(
            num_drones=args.num_drones,
            gui=args.gui,
            safe_distance=0.30,
            collision_distance=0.10,
        )
    )
    vel_low = np.array([-env.cfg.max_target_speed_xy, -env.cfg.max_target_speed_xy, -env.cfg.max_target_speed_z], dtype=np.float32)
    vel_high = np.array([env.cfg.max_target_speed_xy, env.cfg.max_target_speed_xy, env.cfg.max_target_speed_z], dtype=np.float32)

    use_diff_qp = bool(args.diff_qp)
    if use_diff_qp:
        solver = build_solver_from_velocity_bounds(
            num_drones=args.num_drones,
            max_speed_xy=env.cfg.max_target_speed_xy,
            max_speed_z=env.cfg.max_target_speed_z,
            config=DifferentiableCBFQPConfig(
                safe_distance=env.cfg.safe_distance,
                alpha=4.0,
                solve_method="SCS",
                n_jobs_forward=1,
                n_jobs_backward=1,
            ),
            device=args.device,
        )
        numpy_solver = None
    else:
        solver = None
        numpy_solver = NumpyCBFQPSolver(
            num_drones=args.num_drones,
            safe_distance=env.cfg.safe_distance,
            alpha=4.0,
            vel_low=vel_low,
            vel_high=vel_high,
        )

    obs, info = env.reset(seed=args.seed)
    cbf_state = np.asarray(info["cbf_state"], dtype=np.float32)
    obs_dim = int(obs.shape[0])
    action_dim = args.num_drones * 3

    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.lr)

    episode_return = 0.0
    episode_len = 0
    recent_returns: List[float] = []
    recent_min_d: List[float] = []

    for update in range(1, args.total_updates + 1):
        obs_buf = np.zeros((ppo_cfg.rollout_steps, obs_dim), dtype=np.float32)
        cbf_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, 6), dtype=np.float32)
        act_buf = np.zeros((ppo_cfg.rollout_steps, action_dim), dtype=np.float32)
        logp_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        min_d_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        slack_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)

        for t in range(ppo_cfg.rollout_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mean_t, std_t, value_t = model(obs_t)
            dist = Normal(mean_t, std_t)
            u_nom_flat = dist.rsample()
            logp_t = dist.log_prob(u_nom_flat).sum(dim=-1)
            u_nom = u_nom_flat.reshape(args.num_drones, 3)
            u_nom_np = u_nom.detach().cpu().numpy()

            if use_diff_qp:
                cbf_t = torch.as_tensor(cbf_state, dtype=torch.float32, device=device)
                u_safe, slack = solver.solve_torch(cbf_t, u_nom)
                action_np = u_safe.detach().cpu().numpy().astype(np.float32)
                slack_norm = float(np.linalg.norm(slack.detach().cpu().numpy()))
            else:
                action_np, qp_info = numpy_solver(cbf_state, u_nom_np)
                slack_norm = float(qp_info.get("slack_l2", np.nan))

            next_obs, reward, terminated, truncated, next_info = env.step(
                action_np
            )
            done = bool(terminated or truncated)

            obs_buf[t] = obs
            cbf_buf[t] = cbf_state
            act_buf[t] = u_nom_flat.detach().cpu().numpy()[0]
            logp_buf[t] = float(logp_t.detach().cpu().numpy()[0])
            val_buf[t] = float(value_t.detach().cpu().numpy()[0])
            rew_buf[t] = float(reward)
            done_buf[t] = float(done)
            min_d_buf[t] = float(next_info.get("min_pairwise_distance", np.nan))
            slack_buf[t] = slack_norm

            episode_return += reward
            episode_len += 1

            obs = next_obs
            cbf_state = np.asarray(next_info["cbf_state"], dtype=np.float32)

            if done:
                recent_returns.append(episode_return)
                recent_min_d.append(float(np.nanmin(min_d_buf[max(0, t - episode_len + 1) : t + 1])))
                obs, next_info = env.reset()
                cbf_state = np.asarray(next_info["cbf_state"], dtype=np.float32)
                episode_return = 0.0
                episode_len = 0

        with torch.no_grad():
            obs_last_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, last_value_t = model(obs_last_t)
            last_value = float(last_value_t.cpu().numpy()[0])

        adv, ret = compute_gae(
            rewards=rew_buf,
            values=val_buf,
            dones=done_buf,
            last_value=last_value,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(obs_buf, dtype=torch.float32, device=device)
        cbf_t = torch.as_tensor(cbf_buf, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act_buf, dtype=torch.float32, device=device)
        old_logp_t = torch.as_tensor(logp_buf, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)

        last_qp_grad = 0.0
        last_qp_aux = 0.0
        for _ in range(ppo_cfg.epochs):
            for idx in batch_indices(ppo_cfg.rollout_steps, ppo_cfg.minibatch_size):
                b_obs = obs_t[idx]
                b_cbf = cbf_t[idx]
                b_act = act_t[idx]
                b_old_logp = old_logp_t[idx]
                b_ret = ret_t[idx]
                b_adv = adv_t[idx]

                mean_b, std_b, value_b = model(b_obs)
                dist_b = Normal(mean_b, std_b)
                logp_b = dist_b.log_prob(b_act).sum(dim=-1)
                entropy_b = dist_b.entropy().sum(dim=-1).mean()

                ratio = torch.exp(logp_b - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - ppo_cfg.clip_ratio, 1.0 + ppo_cfg.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * ((value_b - b_ret) ** 2).mean()

                # QP gradient penetration probe and auxiliary loss.
                qp_aux = torch.tensor(0.0, dtype=torch.float32, device=device)
                qp_grad_probe = torch.tensor(0.0, dtype=torch.float32, device=device)
                if use_diff_qp:
                    for j in range(b_obs.shape[0]):
                        u_nom_j = mean_b[j].reshape(args.num_drones, 3)
                        cbf_j = b_cbf[j]
                        u_safe_j, slack_j = solver.solve_torch(cbf_j, u_nom_j)
                        qp_aux = qp_aux + torch.mean((u_safe_j - u_nom_j) ** 2) + 0.1 * torch.mean(slack_j**2)

                        u_probe = u_nom_j.detach().clone().requires_grad_(True)
                        u_safe_probe, _ = solver.solve_torch(cbf_j, u_probe)
                        g = torch.autograd.grad(u_safe_probe.sum(), u_probe, retain_graph=False, create_graph=False)[0]
                        qp_grad_probe = qp_grad_probe + torch.mean(torch.abs(g.detach()))
                    qp_aux = qp_aux / b_obs.shape[0]
                    qp_grad_probe = qp_grad_probe / b_obs.shape[0]

                loss = (
                    policy_loss
                    + ppo_cfg.value_coef * value_loss
                    - ppo_cfg.entropy_coef * entropy_b
                    + ppo_cfg.qp_aux_weight * qp_aux
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

                last_qp_grad = float(qp_grad_probe.cpu().item())
                last_qp_aux = float(qp_aux.detach().cpu().item())

        avg_return = float(np.mean(recent_returns[-10:])) if recent_returns else 0.0
        avg_min_d = float(np.nanmean(min_d_buf))
        avg_slack = float(np.mean(slack_buf))
        print(
            f"update={update:04d} avg_return_10={avg_return:.3f} "
            f"rollout_min_d={avg_min_d:.3f} rollout_slack={avg_slack:.4f} "
            f"qp_aux={last_qp_aux:.4f} qp_grad_probe={last_qp_grad:.6f} diff_qp={use_diff_qp}"
        )

    env.close()


if __name__ == "__main__":
    main()
