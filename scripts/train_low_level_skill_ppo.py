import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

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
    LocalObstacleEnvConfig,
    LocalObstacleFormationEnv,
    SkillConditionedActorCritic,
    SkillConditionedPolicyConfig,
    build_solver_from_velocity_bounds,
)


@dataclass
class PPOConfig:
    rollout_steps: int = 256
    epochs: int = 6
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.8
    qp_aux_weight: float = 0.10


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t_len = len(rewards)
    adv = np.zeros(t_len, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(t_len)):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_value if t == t_len - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv, ret


def batch_indices(n: int, batch_size: int):
    perm = np.random.permutation(n)
    for s in range(0, n, batch_size):
        yield perm[s : s + batch_size]


def velocities_to_normalized(v_cmd: np.ndarray, max_xy: float, max_z: float) -> np.ndarray:
    act = np.zeros_like(v_cmd, dtype=np.float32)
    act[:, 0:2] = v_cmd[:, 0:2] / float(max_xy)
    act[:, 2] = v_cmd[:, 2] / float(max_z)
    return np.clip(act, -1.0, 1.0).astype(np.float32)


def parse_skill_set(skill_set_text: str, num_skills: int) -> np.ndarray:
    out: List[int] = []
    for token in skill_set_text.split(","):
        t = token.strip()
        if not t:
            continue
        v = int(t)
        if v < 0 or v >= num_skills:
            raise ValueError(f"Skill id {v} out of range [0, {num_skills-1}]")
        out.append(v)
    if not out:
        raise ValueError("skill-set cannot be empty")
    return np.asarray(sorted(set(out)), dtype=np.int64)


def build_skill_targets(
    desired_positions: np.ndarray,
    skill_idx: np.ndarray,
    delta_x: float,
    delta_y: float,
    delta_z: float,
) -> np.ndarray:
    target = np.asarray(desired_positions, dtype=np.float32).copy()
    z = np.asarray(skill_idx, dtype=np.int64).reshape(-1)
    for i in range(target.shape[0]):
        zi = int(z[i])
        if zi == 1:
            target[i, 1] -= float(delta_y)  # left
        elif zi == 2:
            target[i, 1] += float(delta_y)  # right
        elif zi == 3:
            target[i, 2] += float(delta_z)  # up
        elif zi == 4:
            target[i, 2] -= float(delta_z)  # down
        elif zi == 5:
            target[i, 0] += float(delta_x)  # forward
        elif zi == 6:
            target[i, 0] -= float(delta_x)  # backward
    return target


def build_policy_obs(
    raw_obs: np.ndarray,
    skill_targets: np.ndarray,
    use_delta_feature: bool = True,
) -> np.ndarray:
    obs = np.asarray(raw_obs, dtype=np.float32)
    if not use_delta_feature:
        return obs
    pos = obs[:, 0:3]
    delta = np.asarray(skill_targets, dtype=np.float32) - pos
    dist = np.linalg.norm(delta, axis=1, keepdims=True).astype(np.float32)
    return np.concatenate([obs, delta.astype(np.float32), dist], axis=1).astype(np.float32)


def compute_teacher_velocity(
    pos: np.ndarray,
    vel: np.ndarray,
    target: np.ndarray,
    kp: float,
    kd: float,
    max_xy: float,
    max_z: float,
) -> np.ndarray:
    u = kp * (target - pos) - kd * vel
    u = np.asarray(u, dtype=np.float32)
    u[:, 0:2] = np.clip(u[:, 0:2], -float(max_xy), float(max_xy))
    u[:, 2] = np.clip(u[:, 2], -float(max_z), float(max_z))
    return u.astype(np.float32)


def sample_initial_skill_schedule(
    num_drones: int,
    allowed_skills: np.ndarray,
    k_min: int,
    k_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    skills = np.random.choice(allowed_skills, size=(num_drones,)).astype(np.int64)
    periods = np.random.randint(k_min, k_max + 1, size=(num_drones,), dtype=np.int64)
    ages = np.array([np.random.randint(0, max(int(p), 1)) for p in periods], dtype=np.int64)
    return skills, periods, ages


def main() -> None:
    parser = argparse.ArgumentParser(description="White-box low-level PPO pretraining with random skill commands")
    parser.add_argument("--num-drones", type=int, default=4)
    parser.add_argument("--scenario", type=str, default="bridge_tree", choices=["none", "bridge", "tree", "bridge_tree"])
    parser.add_argument("--total-updates", type=int, default=120)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-skills", type=int, default=7)
    parser.add_argument("--skill-set", type=str, default="0,1,2,3,4")
    parser.add_argument("--use-delta-feature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skill-period-min", type=int, default=40)
    parser.add_argument("--skill-period-max", type=int, default=60)
    parser.add_argument("--skill-stay-prob", type=float, default=0.85)
    parser.add_argument("--delta-x", type=float, default=0.30)
    parser.add_argument("--delta-y", type=float, default=0.45)
    parser.add_argument("--delta-z", type=float, default=0.25)
    parser.add_argument("--skill-target-sigma", type=float, default=0.35)
    parser.add_argument("--reset-err-threshold", type=float, default=3.0)
    parser.add_argument("--reward-clip", type=float, default=3.0)
    parser.add_argument("--teacher-kp", type=float, default=1.2)
    parser.add_argument("--teacher-kd", type=float, default=0.25)
    parser.add_argument("--teacher-loss-coef", type=float, default=0.5)
    parser.add_argument("--teacher-loss-decay-updates", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-len-sec", type=float, default=20.0)
    parser.add_argument("--bridge-offset-y", type=float, default=0.40)
    parser.add_argument("--diff-qp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w-skill", type=float, default=1.0)
    parser.add_argument("--w-smooth", type=float, default=0.10)
    parser.add_argument("--w-intervene", type=float, default=0.10)
    parser.add_argument("--w-slack", type=float, default=0.10)
    parser.add_argument("--w-skill-err", type=float, default=0.05)
    parser.add_argument("--w-speed", type=float, default=0.01)
    parser.add_argument("--w-qp-fail", type=float, default=0.10)
    parser.add_argument("--w-safety", type=float, default=2.0)
    parser.add_argument("--w-contact", type=float, default=6.0)
    parser.add_argument("--w-env", type=float, default=0.0)
    parser.add_argument("--debug-progress-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-progress-scale", type=float, default=10.0)
    parser.add_argument("--debug-goal-threshold", type=float, default=0.10)
    parser.add_argument("--debug-goal-bonus", type=float, default=50.0)
    parser.add_argument(
        "--debug-progress-agent",
        type=int,
        default=-1,
        help="-1 means mean over all drones; otherwise use specific drone index",
    )
    args = parser.parse_args()

    if os.name == "nt" and args.diff_qp:
        raise SystemExit(
            "Windows detected: --diff-qp is unstable with diffcp/cvxpylayers native stack.\n"
            "Please run this script inside WSL/Linux Python, or use --no-diff-qp on Windows."
        )
    if args.skill_period_min <= 0 or args.skill_period_max < args.skill_period_min:
        raise SystemExit("Invalid skill period range")
    if not (0.0 <= args.skill_stay_prob <= 1.0):
        raise SystemExit("skill-stay-prob must be in [0,1]")
    if args.num_skills <= 0:
        raise SystemExit("num-skills must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ppo_cfg = PPOConfig(rollout_steps=args.rollout_steps)
    allowed_skills = parse_skill_set(args.skill_set, args.num_skills)

    env = LocalObstacleFormationEnv(
        LocalObstacleEnvConfig(
            num_drones=args.num_drones,
            gui=args.gui,
            scenario=args.scenario,
            bridge_pillar_offset_y=args.bridge_offset_y,
            episode_len_sec=args.episode_len_sec,
        )
    )
    obs, info = env.reset(seed=args.seed)
    cbf_state = np.asarray(info["cbf_state"], dtype=np.float32)

    obs_dim_raw = int(obs.shape[1])
    obs_dim = obs_dim_raw + (4 if args.use_delta_feature else 0)
    model = SkillConditionedActorCritic(
        obs_local_dim=obs_dim,
        cfg=SkillConditionedPolicyConfig(num_skills=args.num_skills, hidden_dim=256),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.lr)

    qp_solver = None
    if args.diff_qp:
        qp_solver = build_solver_from_velocity_bounds(
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

    current_skills, skill_periods, skill_ages = sample_initial_skill_schedule(
        num_drones=args.num_drones,
        allowed_skills=allowed_skills,
        k_min=args.skill_period_min,
        k_max=args.skill_period_max,
    )
    prev_u_safe = np.zeros((args.num_drones, 3), dtype=np.float32)

    desired_init = np.asarray(info["desired_positions"], dtype=np.float32)
    target_init = build_skill_targets(desired_init, current_skills, args.delta_x, args.delta_y, args.delta_z)
    pos_init = np.asarray(obs[:, 0:3], dtype=np.float32)
    init_err = np.linalg.norm(pos_init - target_init, axis=1)
    last_progress_dist_n = init_err.astype(np.float32).copy()

    ep_return = 0.0
    recent_returns: List[float] = []

    for update in range(1, args.total_updates + 1):
        obs_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, obs_dim), dtype=np.float32)
        skill_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones), dtype=np.int64)
        tau_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, 1), dtype=np.float32)
        cbf_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, 6), dtype=np.float32)
        u_nom_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, 3), dtype=np.float32)
        teacher_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, 3), dtype=np.float32)
        logp_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        skill_score_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        skill_prog_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        skill_err_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        slack_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        contact_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        intervene_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        qp_fail_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        teacher_mse_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        debug_dist_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        debug_reward_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        switch_count = 0

        for t in range(ppo_cfg.rollout_steps):
            # Asynchronous per-agent skill schedule with persistence.
            switched_mask = np.zeros((args.num_drones,), dtype=bool)
            for i in range(args.num_drones):
                if skill_ages[i] >= skill_periods[i]:
                    if np.random.rand() > args.skill_stay_prob:
                        current_skills[i] = int(np.random.choice(allowed_skills))
                    skill_periods[i] = int(np.random.randint(args.skill_period_min, args.skill_period_max + 1))
                    skill_ages[i] = 0
                    switched_mask[i] = True
                    switch_count += 1

            tau_vec = (skill_ages / np.maximum(skill_periods - 1, 1)).astype(np.float32)

            desired_now = np.asarray(info["desired_positions"], dtype=np.float32)
            target_now = build_skill_targets(
                desired_now, current_skills, args.delta_x, args.delta_y, args.delta_z
            )
            policy_obs = build_policy_obs(obs, target_now, use_delta_feature=args.use_delta_feature)

            obs_t = torch.as_tensor(policy_obs[None, ...], dtype=torch.float32, device=device)
            skill_t = torch.as_tensor(current_skills[None, ...], dtype=torch.long, device=device)
            tau_t = torch.as_tensor(tau_vec[None, :, None], dtype=torch.float32, device=device)
            cbf_t = torch.as_tensor(cbf_state[None, ...], dtype=torch.float32, device=device)

            with torch.no_grad():
                out = model.act(obs_t, skill_t, tau_t, cbf_t, qp_solver=qp_solver, deterministic=False)

            u_nom = out["u_nom"][0].cpu().numpy().astype(np.float32)
            u_safe = out["u_safe"][0].cpu().numpy().astype(np.float32)
            logp_joint = float(out["logp_joint"][0].cpu().numpy())
            value_joint = float(out["value_joint"][0].cpu().numpy())
            slack_aux = float(out["slack_aux"][0].cpu().numpy()) if args.diff_qp else 0.0
            qp_fail_count = int(out.get("qp_fail_count", torch.zeros((1,), dtype=torch.int32))[0].cpu().item())
            qp_fail_ratio = float(qp_fail_count) / float(args.num_drones)

            action = velocities_to_normalized(
                u_safe,
                max_xy=env.cfg.max_target_speed_xy,
                max_z=env.cfg.max_target_speed_z,
            )
            next_obs, env_reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

            desired_next = np.asarray(next_info["desired_positions"], dtype=np.float32)
            pos_now = np.asarray(obs[:, 0:3], dtype=np.float32)
            pos_next = np.asarray(next_obs[:, 0:3], dtype=np.float32)
            target_next = build_skill_targets(
                desired_next, current_skills, args.delta_x, args.delta_y, args.delta_z
            )
            err_now = np.linalg.norm(pos_now - target_now, axis=1)
            err_next = np.linalg.norm(pos_next - target_next, axis=1)
            skill_progress = float(np.mean(np.clip(err_now - err_next, -0.2, 0.2)))
            mean_err_next = float(np.mean(err_next))
            skill_target = float(np.exp(-(mean_err_next**2) / (2.0 * args.skill_target_sigma**2)))
            skill_score = 1.5 * skill_progress + skill_target

            pair_margin = float(next_info["min_pairwise_distance"] - env.cfg.safe_distance)
            obs_clear = float(next_info["min_obstacle_clearance"])
            obs_violation = max(0.0, -obs_clear) if np.isfinite(obs_clear) else 0.0
            safety_violation = max(0.0, -pair_margin) + obs_violation
            contact = float(int(next_info.get("obstacle_contact_count", 0) > 0))
            smooth_pen = float(np.mean((u_safe - prev_u_safe) ** 2))
            intervene_pen = float(np.mean((u_safe - u_nom) ** 2))
            speed_pen = float(np.mean(np.sum(u_safe**2, axis=1)))

            reward = (
                args.w_env * float(env_reward)
                + args.w_skill * skill_score
                - args.w_skill_err * mean_err_next
                - args.w_smooth * smooth_pen
                - args.w_intervene * intervene_pen
                - args.w_slack * slack_aux
                - args.w_speed * speed_pen
                - args.w_qp_fail * qp_fail_ratio
                - args.w_safety * safety_violation
                - args.w_contact * contact
            )
            if args.reward_clip > 0.0:
                reward = float(np.clip(reward, -args.reward_clip, args.reward_clip))

            if args.debug_progress_reward:
                if np.any(switched_mask):
                    # Reset baseline for switched agents to avoid fake negative rewards
                    # from target discontinuities.
                    last_progress_dist_n[switched_mask] = err_now[switched_mask]

                if args.debug_progress_agent >= 0 and args.debug_progress_agent < args.num_drones:
                    idx = int(args.debug_progress_agent)
                    current_dist = float(err_next[idx])
                    reward = (float(last_progress_dist_n[idx]) - current_dist) * float(args.debug_progress_scale)
                    last_progress_dist_n[idx] = float(current_dist)
                else:
                    progress_vec = last_progress_dist_n - err_next
                    reward = float(np.mean(progress_vec)) * float(args.debug_progress_scale)
                    current_dist = float(np.mean(err_next))
                    last_progress_dist_n[:] = err_next.astype(np.float32)

                if current_dist < float(args.debug_goal_threshold):
                    reward += float(args.debug_goal_bonus)
                    done = True
                debug_dist_buf[t] = current_dist
                debug_reward_buf[t] = float(reward)

            obs_buf[t] = policy_obs
            skill_buf[t] = current_skills
            tau_buf[t, :, 0] = tau_vec
            cbf_buf[t] = cbf_state
            u_nom_buf[t] = u_nom
            logp_buf[t] = logp_joint
            val_buf[t] = value_joint
            rew_buf[t] = reward
            done_buf[t] = float(done)
            skill_score_buf[t] = skill_score
            skill_prog_buf[t] = skill_progress
            skill_err_buf[t] = mean_err_next
            slack_buf[t] = slack_aux
            contact_buf[t] = contact
            intervene_buf[t] = intervene_pen
            qp_fail_buf[t] = qp_fail_ratio

            ep_return += reward
            obs = next_obs
            info = next_info
            cbf_state = np.asarray(next_info["cbf_state"], dtype=np.float32)
            prev_u_safe = u_safe
            skill_ages += 1

            if mean_err_next > args.reset_err_threshold:
                done = True

            if done:
                recent_returns.append(ep_return)
                ep_return = 0.0
                obs, info = env.reset()
                cbf_state = np.asarray(info["cbf_state"], dtype=np.float32)
                prev_u_safe[:] = 0.0
                current_skills, skill_periods, skill_ages = sample_initial_skill_schedule(
                    num_drones=args.num_drones,
                    allowed_skills=allowed_skills,
                    k_min=args.skill_period_min,
                    k_max=args.skill_period_max,
                )
                desired_reset = np.asarray(info["desired_positions"], dtype=np.float32)
                target_reset = build_skill_targets(
                    desired_reset, current_skills, args.delta_x, args.delta_y, args.delta_z
                )
                pos_reset = np.asarray(obs[:, 0:3], dtype=np.float32)
                err_reset = np.linalg.norm(pos_reset - target_reset, axis=1)
                last_progress_dist_n[:] = err_reset.astype(np.float32)

        with torch.no_grad():
            tau_vec = (skill_ages / np.maximum(skill_periods - 1, 1)).astype(np.float32)
            desired_last = np.asarray(info["desired_positions"], dtype=np.float32)
            target_last = build_skill_targets(
                desired_last, current_skills, args.delta_x, args.delta_y, args.delta_z
            )
            policy_obs_last = build_policy_obs(obs, target_last, use_delta_feature=args.use_delta_feature)
            obs_t = torch.as_tensor(policy_obs_last[None, ...], dtype=torch.float32, device=device)
            skill_t = torch.as_tensor(current_skills[None, ...], dtype=torch.long, device=device)
            tau_t = torch.as_tensor(tau_vec[None, :, None], dtype=torch.float32, device=device)
            _, _, value_last_agent = model.forward(obs_t, skill_t, tau_t)
            last_value = float(value_last_agent.mean(dim=-1).cpu().numpy()[0])

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
        skill_t = torch.as_tensor(skill_buf, dtype=torch.long, device=device)
        tau_t = torch.as_tensor(tau_buf, dtype=torch.float32, device=device)
        cbf_t = torch.as_tensor(cbf_buf, dtype=torch.float32, device=device)
        u_nom_t = torch.as_tensor(u_nom_buf, dtype=torch.float32, device=device)
        old_logp_t = torch.as_tensor(logp_buf, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)

        last_qp_aux = 0.0
        for _ in range(ppo_cfg.epochs):
            for idx in batch_indices(ppo_cfg.rollout_steps, ppo_cfg.minibatch_size):
                b_obs = obs_t[idx]
                b_skill = skill_t[idx]
                b_tau = tau_t[idx]
                b_cbf = cbf_t[idx]
                b_nom = u_nom_t[idx]
                b_old_logp = old_logp_t[idx]
                b_ret = ret_t[idx]
                b_adv = adv_t[idx]

                mean_b, std_b, value_agent_b = model.forward(b_obs, b_skill, b_tau)
                value_b = value_agent_b.mean(dim=-1)
                dist_b = Normal(mean_b, std_b)
                logp_b = dist_b.log_prob(b_nom).sum(dim=-1).sum(dim=-1)
                entropy_b = dist_b.entropy().sum(dim=-1).sum(dim=-1).mean()

                ratio = torch.exp(logp_b - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - ppo_cfg.clip_ratio, 1.0 + ppo_cfg.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((value_b - b_ret) ** 2).mean()

                qp_aux = torch.tensor(0.0, dtype=torch.float32, device=device)
                if args.diff_qp and qp_solver is not None:
                    safe_list = []
                    slack_list = []
                    for j in range(mean_b.shape[0]):
                        try:
                            u_safe_j, slack_j = qp_solver.solve_torch(b_cbf[j], mean_b[j])
                        except Exception:
                            u_safe_j = mean_b[j]
                            slack_j = torch.zeros((1,), dtype=mean_b.dtype, device=mean_b.device)
                        safe_list.append(u_safe_j)
                        slack_list.append(torch.mean(slack_j**2))
                    safe_stack = torch.stack(safe_list, dim=0)
                    slack_stack = torch.stack(slack_list, dim=0)
                    qp_aux = torch.mean((safe_stack - mean_b) ** 2) + 0.1 * torch.mean(slack_stack)

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
                last_qp_aux = float(qp_aux.detach().cpu().item())

        avg_return = float(np.mean(recent_returns[-10:])) if recent_returns else 0.0
        print(
            f"update={update:04d} avg_return_10={avg_return:.3f} "
            f"rollout_skill_score={float(np.mean(skill_score_buf)):.3f} "
            f"rollout_skill_progress={float(np.mean(skill_prog_buf)):.4f} "
            f"rollout_skill_err={float(np.mean(skill_err_buf)):.3f} "
            f"rollout_intervene={float(np.mean(intervene_buf)):.4f} "
            f"rollout_slack={float(np.mean(slack_buf)):.4f} "
            f"rollout_qp_fail={float(np.mean(qp_fail_buf)):.4f} "
            f"rollout_contact={float(np.mean(contact_buf)):.4f} "
            f"switches={switch_count} qp_aux={last_qp_aux:.4f} diff_qp={args.diff_qp} "
            f"debug_mode={args.debug_progress_reward} "
            f"debug_dist={float(np.mean(debug_dist_buf)):.3f} "
            f"debug_reward={float(np.mean(debug_reward_buf)):.3f}"
        )

    env.close()


if __name__ == "__main__":
    main()
