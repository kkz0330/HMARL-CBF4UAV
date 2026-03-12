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
    skill_velocity_alignment_reward,
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
    next_values: np.ndarray,
    episode_dones: np.ndarray,
    terminals: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t_len = len(rewards)
    adv = np.zeros(t_len, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(t_len)):
        next_nonterminal = 1.0 - terminals[t]
        next_value = next_values[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        # Cut GAE recursion at any episode boundary, but keep timeout bootstrap.
        last_gae = delta + gamma * gae_lambda * (1.0 - episode_dones[t]) * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv, ret


def batch_indices(n: int, batch_size: int):
    perm = np.random.permutation(n)
    for s in range(0, n, batch_size):
        yield perm[s : s + batch_size]


def linear_anneal_weight(
    update: int,
    total_updates: int,
    start_weight: float,
    end_weight: float,
    start_frac: float,
    end_frac: float,
) -> float:
    total = max(int(total_updates), 1)
    s = float(np.clip(start_frac, 0.0, 1.0))
    e = float(np.clip(end_frac, 0.0, 1.0))
    if e < s:
        e = s

    if total <= 1:
        alpha = 1.0
    else:
        p = (float(update) - 1.0) / float(total - 1)
        if p <= s:
            alpha = 0.0
        elif p >= e:
            alpha = 1.0
        else:
            alpha = (p - s) / max(e - s, 1e-8)
    return float((1.0 - alpha) * float(start_weight) + alpha * float(end_weight))


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
            # accelerate: no positional offset, handled by speed CLF
            pass
        elif zi == 6:
            # decelerate: no positional offset, handled by speed CLF
            pass
    return target


def build_skill_reference_velocity(
    skill_idx: np.ndarray,
    cruise_speed: float,
    accelerate_speed: float,
    decelerate_speed: float,
    vel_now_n3: np.ndarray | None = None,
    speed_cap: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference heading (unit vector) and speed per skill.

    For speed skills:
      - cruise(0): target speed = cruise_speed
      - accelerate(5): target speed = clip(vx_now + accelerate_speed, 0, speed_cap)
      - decelerate(6): target speed = clip(vx_now - decelerate_speed, 0, speed_cap)
    """
    z = np.asarray(skill_idx, dtype=np.int64).reshape(-1)
    n = z.shape[0]
    if vel_now_n3 is None:
        vel_now = np.zeros((n, 3), dtype=np.float32)
    else:
        vel_now = np.asarray(vel_now_n3, dtype=np.float32)
        if vel_now.shape != (n, 3):
            raise ValueError(f"vel_now_n3 must have shape {(n, 3)}, got {vel_now.shape}")

    spd_cap = float("inf") if speed_cap is None else float(max(speed_cap, 0.0))
    accel_delta = float(max(accelerate_speed, 0.0))
    decel_delta = float(max(decelerate_speed, 0.0))

    ref_dir = np.zeros((n, 3), dtype=np.float32)
    ref_speed = np.full((n,), float(cruise_speed), dtype=np.float32)
    for i in range(n):
        zi = int(z[i])
        if zi == 1:
            ref_dir[i] = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # left
        elif zi == 2:
            ref_dir[i] = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # right
        elif zi == 3:
            ref_dir[i] = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # up
        elif zi == 4:
            ref_dir[i] = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # down
        else:
            ref_dir[i] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # cruise/accelerate/decelerate

        if zi == 5:
            base = max(float(vel_now[i, 0]), 0.0)
            ref_speed[i] = float(np.clip(base + accel_delta, 0.0, spd_cap))
        elif zi == 6:
            base = max(float(vel_now[i, 0]), 0.0)
            ref_speed[i] = float(np.clip(base - decel_delta, 0.0, spd_cap))
        else:
            ref_speed[i] = float(cruise_speed)
    return ref_dir, ref_speed


def in_skill_initiation_set(obs_i: np.ndarray, min_altitude: float = 0.2) -> bool:
    """Initiation set I_z(o): basic state validity for starting a new skill."""
    x = np.asarray(obs_i, dtype=np.float32).reshape(-1)
    if x.size < 3 or not np.isfinite(x).all():
        return False
    return bool(float(x[2]) >= float(min_altitude))


def evaluate_skill_termination(
    pos_n3: np.ndarray,
    vel_n3: np.ndarray,
    target_n3: np.ndarray,
    skill_idx_n: np.ndarray,
    age_after_n: np.ndarray,
    t_max_n: np.ndarray,
    ref_dir_n3: np.ndarray,
    ref_speed_n: np.ndarray,
    pos_tol: float,
    speed_tol: float,
    heading_tol: float,
    min_duration: int,
    pos_guard_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Termination function Phi_z(o,t): success in J_z or timeout T_max."""
    pos = np.asarray(pos_n3, dtype=np.float32)
    vel = np.asarray(vel_n3, dtype=np.float32)
    target = np.asarray(target_n3, dtype=np.float32)
    z = np.asarray(skill_idx_n, dtype=np.int64).reshape(-1)
    age_after = np.asarray(age_after_n, dtype=np.int64).reshape(-1)
    t_max = np.asarray(t_max_n, dtype=np.int64).reshape(-1)
    ref_dir = np.asarray(ref_dir_n3, dtype=np.float32)
    ref_speed = np.asarray(ref_speed_n, dtype=np.float32).reshape(-1)

    n = z.shape[0]
    term = np.zeros((n,), dtype=bool)
    succ = np.zeros((n,), dtype=bool)
    tout = np.zeros((n,), dtype=bool)

    pos_tol = float(max(pos_tol, 1e-4))
    speed_tol = float(max(speed_tol, 1e-4))
    heading_tol = float(max(heading_tol, 1e-4))
    min_duration = int(max(min_duration, 0))
    pos_guard = pos_tol * float(max(pos_guard_scale, 1.0))

    for i in range(n):
        if int(age_after[i]) < min_duration:
            continue

        timeout_i = bool(int(age_after[i]) >= int(t_max[i]))
        if timeout_i:
            term[i] = True
            tout[i] = True
            continue

        pos_err = float(np.linalg.norm(pos[i] - target[i]))
        speed = float(np.linalg.norm(vel[i]))
        zi = int(z[i])

        if zi in (1, 2, 3, 4):
            # Lateral/vertical skills: terminate on reaching offset target and slowing down.
            success_i = (pos_err <= pos_tol) and (speed <= speed_tol)
        else:
            # Cruise/accel/decel: terminate on speed/heading tracking and no large position drift.
            speed_err = abs(speed - float(ref_speed[i]))
            vxy = vel[i, 0:2]
            dxy = ref_dir[i, 0:2]
            nv = float(np.linalg.norm(vxy))
            nd = float(np.linalg.norm(dxy))
            if nv < 1e-4 or nd < 1e-4:
                heading_ok = speed <= speed_tol
            else:
                cos_h = float(np.clip(np.dot(vxy, dxy) / (nv * nd + 1e-6), -1.0, 1.0))
                heading_err = float(np.arccos(cos_h))
                heading_ok = heading_err <= heading_tol
            success_i = (speed_err <= speed_tol) and heading_ok and (pos_err <= pos_guard)

        if success_i:
            term[i] = True
            succ[i] = True

    return term, succ, tout


def build_policy_obs(
    raw_obs: np.ndarray,
    skill_targets: np.ndarray,
    use_delta_feature: bool = True,
    normalize_obs: bool = True,
    num_drones: int = 1,
    max_sensed_obstacles: int = 3,
    include_neighbor_features: bool = True,
    pos_scale: float = 5.0,
    vel_scale: float = 2.0,
    clip_val: float = 5.0,
) -> np.ndarray:
    raw = np.asarray(raw_obs, dtype=np.float32)
    obs = raw.copy()
    if normalize_obs:
        pos_scale = max(float(pos_scale), 1e-6)
        vel_scale = max(float(vel_scale), 1e-6)
        for i in range(int(num_drones)):
            k = 0
            obs[i, k : k + 3] /= pos_scale
            k += 3
            obs[i, k : k + 3] /= vel_scale
            k += 3
            obs[i, k : k + 3] /= pos_scale
            k += 3

            if include_neighbor_features:
                for _ in range(int(num_drones) - 1):
                    obs[i, k : k + 3] /= pos_scale
                    k += 3
                    obs[i, k] /= pos_scale
                    k += 1

            for _ in range(int(max_sensed_obstacles)):
                obs[i, k : k + 3] /= pos_scale
                k += 3
                obs[i, k] /= pos_scale
                k += 1
            if k != obs.shape[1]:
                raise ValueError(f"Unexpected observation shape: expected per-drone dim {k}, got {obs.shape[1]}")

        if clip_val > 0.0:
            obs = np.clip(obs, -float(clip_val), float(clip_val))

    if not use_delta_feature:
        return obs.astype(np.float32)

    pos = raw[:, 0:3]
    delta = np.asarray(skill_targets, dtype=np.float32) - pos
    dist = np.linalg.norm(delta, axis=1, keepdims=True).astype(np.float32)
    if normalize_obs:
        scale = max(float(pos_scale), 1e-6)
        delta = delta / scale
        dist = dist / scale
        if clip_val > 0.0:
            delta = np.clip(delta, -float(clip_val), float(clip_val))
            dist = np.clip(dist, -float(clip_val), float(clip_val))
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
    parser.add_argument("--scenario", type=str, default="single_pillar", choices=["none", "bridge", "tree", "bridge_tree", "single_pillar"])
    parser.add_argument("--formation-pattern", type=str, default="line", choices=["line", "square", "auto"])
    parser.add_argument("--formation-spacing", type=float, default=0.3)
    parser.add_argument("--use-moving-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--goal-start-x", type=float, default=0.0)
    parser.add_argument("--goal-start-y", type=float, default=0.0)
    parser.add_argument("--goal-start-z", type=float, default=1.0)
    parser.add_argument("--goal-end-x", type=float, default=4.0)
    parser.add_argument("--goal-end-y", type=float, default=0.0)
    parser.add_argument("--goal-end-z", type=float, default=1.0)
    parser.add_argument("--goal-speed", type=float, default=0.5)
    parser.add_argument("--single-pillar-x", type=float, default=2.0)
    parser.add_argument("--single-pillar-y", type=float, default=0.0)
    parser.add_argument("--fall-z-threshold", type=float, default=0.15)
    parser.add_argument("--fall-penalty", type=float, default=10.0)
    parser.add_argument("--terminate-on-fall", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--total-updates", type=int, default=120)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-skills", type=int, default=7)
    parser.add_argument("--skill-set", type=str, default="0,1,2,3,4,5,6")
    parser.add_argument("--use-delta-feature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-pos-scale", type=float, default=5.0)
    parser.add_argument("--obs-vel-scale", type=float, default=2.0)
    parser.add_argument("--obs-clip", type=float, default=5.0)
    parser.add_argument("--skill-period-min", type=int, default=40)
    parser.add_argument("--skill-period-max", type=int, default=60)
    parser.add_argument("--skill-stay-prob", type=float, default=0.85)
    parser.add_argument("--skill-init-min-alt", type=float, default=0.2)
    parser.add_argument("--skill-min-duration", type=int, default=8)
    parser.add_argument("--skill-term-pos-tol", type=float, default=0.18)
    parser.add_argument("--skill-term-speed-tol", type=float, default=0.18)
    parser.add_argument("--skill-term-heading-tol", type=float, default=0.40)
    parser.add_argument("--skill-term-pos-guard-scale", type=float, default=3.0)
    parser.add_argument(
        "--lock-skill-per-episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, keep each drone's sampled skill fixed for the whole episode.",
    )
    parser.add_argument("--delta-x", type=float, default=0.30)
    parser.add_argument("--delta-y", type=float, default=0.45)
    parser.add_argument("--delta-z", type=float, default=0.25)
    parser.add_argument("--skill-target-sigma", type=float, default=0.35)
    parser.add_argument("--reset-err-threshold", type=float, default=3.0)
    parser.add_argument("--reward-clip", type=float, default=0.0)
    parser.add_argument("--teacher-kp", type=float, default=1.2)
    parser.add_argument("--teacher-kd", type=float, default=0.25)
    parser.add_argument("--teacher-loss-coef", type=float, default=0.5)
    parser.add_argument("--teacher-loss-decay-updates", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-level-qp-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--episode-len-sec", type=float, default=20.0)
    parser.add_argument("--bridge-offset-y", type=float, default=0.40)
    parser.add_argument("--diff-qp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qp-enable-clf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qp-clf-mode", type=str, default="skill", choices=["position", "skill"])
    parser.add_argument("--qp-clf-rate", type=float, default=1.0)
    parser.add_argument("--qp-heading-clf-rate", type=float, default=2.0)
    parser.add_argument("--qp-clf-deadzone", type=float, default=0.05)
    parser.add_argument("--qp-speed-clf-deadzone", type=float, default=0.08)
    parser.add_argument("--qp-heading-clf-deadzone", type=float, default=0.08)
    parser.add_argument("--qp-clf-slack-weight", type=float, default=80.0)
    parser.add_argument("--skill-cruise-speed", type=float, default=0.5)
    parser.add_argument("--skill-accelerate-speed", type=float, default=0.2, help="Forward speed increment for accelerate skill (m/s).")
    parser.add_argument("--skill-decelerate-speed", type=float, default=0.2, help="Forward speed decrement for decelerate skill (m/s).")
    parser.add_argument("--use-parametric-qp-objective", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qp-h-base", type=float, default=2.0)
    parser.add_argument("--qp-h-min", type=float, default=1e-3)
    parser.add_argument("--qp-f-scale", type=float, default=0.5)
    parser.add_argument("--u-nom-anchor-start", type=float, default=1.0)
    parser.add_argument("--u-nom-anchor-end", type=float, default=0.0)
    parser.add_argument("--u-nom-anchor-anneal-start-frac", type=float, default=0.0)
    parser.add_argument("--u-nom-anchor-anneal-end-frac", type=float, default=1.0)
    parser.add_argument("--w-skill", type=float, default=1.0)
    parser.add_argument("--w-align", type=float, default=0.8)
    parser.add_argument("--align-distance-gating", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-gate-near", type=float, default=0.25)
    parser.add_argument("--align-gate-far", type=float, default=1.20)
    parser.add_argument("--align-gate-min", type=float, default=0.0)
    parser.add_argument("--w-smooth", type=float, default=0.10)
    parser.add_argument("--w-intervene", type=float, default=0.02)
    parser.add_argument("--w-slack", type=float, default=0.50)
    parser.add_argument("--w-skill-err", type=float, default=0.05)
    parser.add_argument("--w-speed", type=float, default=0.005)
    parser.add_argument("--w-qp-fail", type=float, default=0.10)
    parser.add_argument("--w-safety", type=float, default=2.0)
    parser.add_argument("--w-contact", type=float, default=6.0)
    parser.add_argument("--w-env", type=float, default=0.0)
    parser.add_argument("--c1-action", type=float, default=0.05)
    parser.add_argument("--c2-vel", type=float, default=0.60)
    parser.add_argument("--c3-heading", type=float, default=0.25)
    parser.add_argument("--c4-pos", type=float, default=0.35)
    parser.add_argument("--v-des-eps", type=float, default=0.10)
    parser.add_argument("--heading-speed-min", type=float, default=0.05)
    parser.add_argument("--w-progress", type=float, default=0.25)
    parser.add_argument("--w-accel-pen", type=float, default=0.15)
    parser.add_argument("--w-turn-pen", type=float, default=0.20)
    parser.add_argument("--w-speed-ref-pen", type=float, default=0.45)
    parser.add_argument("--w-heading-ref-pen", type=float, default=0.40)
    parser.add_argument("--w-pos-pen", type=float, default=0.20)
    parser.add_argument("--debug-progress-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-progress-scale", type=float, default=10.0)
    parser.add_argument("--debug-goal-threshold", type=float, default=0.10)
    parser.add_argument("--debug-goal-bonus", type=float, default=50.0)
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--save-every", type=int, default=20)
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
    if args.skill_min_duration < 0:
        raise SystemExit("skill-min-duration must be >= 0")
    if args.skill_term_pos_tol <= 0.0 or args.skill_term_speed_tol <= 0.0 or args.skill_term_heading_tol <= 0.0:
        raise SystemExit("skill termination tolerances must be > 0")
    if args.skill_cruise_speed < 0.0 or args.skill_accelerate_speed < 0.0 or args.skill_decelerate_speed < 0.0:
        raise SystemExit("skill speeds must be non-negative")
    if args.align_gate_far <= args.align_gate_near:
        raise SystemExit("align-gate-far must be larger than align-gate-near")
    if not (0.0 <= args.align_gate_min <= 1.0):
        raise SystemExit("align-gate-min must be in [0,1]")
    if args.fall_z_threshold < 0.0 or args.fall_penalty < 0.0:
        raise SystemExit("fall-z-threshold and fall-penalty must be non-negative")
    if args.goal_speed < 0.0:
        raise SystemExit("goal-speed must be non-negative")
    if args.u_nom_anchor_anneal_start_frac < 0.0 or args.u_nom_anchor_anneal_end_frac > 1.0:
        raise SystemExit("u-nom-anchor anneal fractions must be within [0,1]")
    if args.u_nom_anchor_anneal_end_frac < args.u_nom_anchor_anneal_start_frac:
        raise SystemExit("u-nom-anchor-anneal-end-frac must be >= start-frac")

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
            formation_pattern=args.formation_pattern,
            formation_spacing=args.formation_spacing,
            use_moving_goal=args.use_moving_goal,
            goal_start_x=args.goal_start_x,
            goal_start_y=args.goal_start_y,
            goal_start_z=args.goal_start_z,
            goal_end_x=args.goal_end_x,
            goal_end_y=args.goal_end_y,
            goal_end_z=args.goal_end_z,
            goal_speed=args.goal_speed,
            single_pillar_x=args.single_pillar_x,
            single_pillar_y=args.single_pillar_y,
            bridge_pillar_offset_y=args.bridge_offset_y,
            episode_len_sec=args.episode_len_sec,
            fall_z_threshold=args.fall_z_threshold,
            fall_penalty=args.fall_penalty,
            terminate_on_fall=args.terminate_on_fall,
        )
    )
    obs, info = env.reset(seed=args.seed)
    cbf_state = np.asarray(info["cbf_state"], dtype=np.float32)

    obs_dim_raw = int(obs.shape[1])
    obs_dim = obs_dim_raw + (4 if args.use_delta_feature else 0)
    model = SkillConditionedActorCritic(
        obs_local_dim=obs_dim,
        cfg=SkillConditionedPolicyConfig(
            num_skills=args.num_skills,
            hidden_dim=256,
            action_scale_xy=env.cfg.max_target_speed_xy,
            action_scale_z=env.cfg.max_target_speed_z,
            use_parametric_qp=args.use_parametric_qp_objective,
            qp_h_base=args.qp_h_base,
            qp_h_min=args.qp_h_min,
            qp_f_scale=args.qp_f_scale,
            qp_only=args.low_level_qp_only,
        ),
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
                enable_clf=args.qp_enable_clf,
                clf_mode=args.qp_clf_mode,
                clf_rate=args.qp_clf_rate,
                heading_clf_rate=args.qp_heading_clf_rate,
                clf_deadzone=args.qp_clf_deadzone,
                speed_clf_deadzone=args.qp_speed_clf_deadzone,
                heading_clf_deadzone=args.qp_heading_clf_deadzone,
                clf_slack_weight=args.qp_clf_slack_weight,
                qp_h_min=args.qp_h_min,
                cruise_speed=args.skill_cruise_speed,
                accelerate_speed=args.skill_accelerate_speed,
                decelerate_speed=args.skill_decelerate_speed,
                skill_speed_cap=env.cfg.max_target_speed_xy,
                max_obstacle_constraints_per_drone=env.cfg.max_sensed_obstacles,
            ),
            device=args.device,
        )

    current_skills, skill_periods, skill_ages = sample_initial_skill_schedule(
        num_drones=args.num_drones,
        allowed_skills=allowed_skills,
        k_min=args.skill_period_min,
        k_max=args.skill_period_max,
    )
    for i in range(args.num_drones):
        if not in_skill_initiation_set(obs[i], min_altitude=args.skill_init_min_alt):
            current_skills[i] = 0
    prev_u_safe = np.zeros((args.num_drones, 3), dtype=np.float32)
    switched_mask_next = np.zeros((args.num_drones,), dtype=bool)

    desired_init = np.asarray(info["desired_positions"], dtype=np.float32)
    target_init = build_skill_targets(desired_init, current_skills, args.delta_x, args.delta_y, args.delta_z)
    pos_init = np.asarray(obs[:, 0:3], dtype=np.float32)
    init_err = np.linalg.norm(pos_init - target_init, axis=1)
    last_progress_dist_n = init_err.astype(np.float32).copy()

    ep_return = 0.0
    recent_returns: List[float] = []

    for update in range(1, args.total_updates + 1):
        u_nom_anchor_w = linear_anneal_weight(
            update=update,
            total_updates=args.total_updates,
            start_weight=args.u_nom_anchor_start,
            end_weight=args.u_nom_anchor_end,
            start_frac=args.u_nom_anchor_anneal_start_frac,
            end_frac=args.u_nom_anchor_anneal_end_frac,
        )
        if args.low_level_qp_only:
            u_nom_anchor_w = 0.0
        obs_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, obs_dim), dtype=np.float32)
        skill_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones), dtype=np.int64)
        tau_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, 1), dtype=np.float32)
        cbf_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, 6), dtype=np.float32)
        obs_cbf_A_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, env.cfg.max_sensed_obstacles, 3), dtype=np.float32)
        obs_cbf_b_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, env.cfg.max_sensed_obstacles), dtype=np.float32)
        target_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, 3), dtype=np.float32)
        u_nom_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, 3), dtype=np.float32)
        teacher_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones, 3), dtype=np.float32)
        logp_buf = np.zeros((ppo_cfg.rollout_steps, args.num_drones), dtype=np.float32)
        val_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        terminal_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        next_val_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        skill_score_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        align_raw_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        align_gate_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        align_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        intrinsic_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        action_pen_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        vel_rel_pen_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        heading_pen_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        pos_sq_pen_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        skill_prog_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        skill_err_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        slack_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        contact_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        intervene_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        qp_fail_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        teacher_mse_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        debug_dist_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        debug_reward_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        term_success_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        term_timeout_buf = np.zeros((ppo_cfg.rollout_steps,), dtype=np.float32)
        switch_count = 0

        for t in range(ppo_cfg.rollout_steps):
            # Asynchronous skill update from previous-step termination Phi_z(o,t).
            switched_mask = switched_mask_next.copy()
            switched_mask_next[:] = False

            tau_vec = np.clip(skill_ages / np.maximum(skill_periods - 1, 1), 0.0, 1.0).astype(np.float32)

            desired_now = np.asarray(info["desired_positions"], dtype=np.float32)
            target_now = build_skill_targets(
                desired_now, current_skills, args.delta_x, args.delta_y, args.delta_z
            )
            policy_obs = build_policy_obs(
                obs,
                target_now,
                use_delta_feature=args.use_delta_feature,
                normalize_obs=args.normalize_obs,
                num_drones=args.num_drones,
                max_sensed_obstacles=env.cfg.max_sensed_obstacles,
                include_neighbor_features=env.cfg.include_neighbor_features,
                pos_scale=args.obs_pos_scale,
                vel_scale=args.obs_vel_scale,
                clip_val=args.obs_clip,
            )

            obs_t = torch.as_tensor(policy_obs[None, ...], dtype=torch.float32, device=device)
            skill_t = torch.as_tensor(current_skills[None, ...], dtype=torch.long, device=device)
            tau_t = torch.as_tensor(tau_vec[None, :, None], dtype=torch.float32, device=device)
            cbf_t = torch.as_tensor(cbf_state[None, ...], dtype=torch.float32, device=device)
            target_t = torch.as_tensor(target_now[None, ...], dtype=torch.float32, device=device)
            target_vel_t = torch.zeros_like(target_t)
            obs_A_now = np.asarray(
                info.get("obstacle_cbf_A", np.zeros((args.num_drones, env.cfg.max_sensed_obstacles, 3), dtype=np.float32)),
                dtype=np.float32,
            )
            obs_b_now = np.asarray(
                info.get("obstacle_cbf_b", np.zeros((args.num_drones, env.cfg.max_sensed_obstacles), dtype=np.float32)),
                dtype=np.float32,
            )
            obs_A_t = torch.as_tensor(obs_A_now[None, ...], dtype=torch.float32, device=device)
            obs_b_t = torch.as_tensor(obs_b_now[None, ...], dtype=torch.float32, device=device)

            with torch.no_grad():
                out = model.act(
                    obs_t,
                    skill_t,
                    tau_t,
                    cbf_t,
                    qp_target_pos=target_t,
                    qp_target_vel=target_vel_t,
                    qp_skill_idx=skill_t,
                    qp_obstacle_A=obs_A_t,
                    qp_obstacle_b=obs_b_t,
                    qp_solver=qp_solver,
                    qp_nominal_anchor_weight=u_nom_anchor_w,
                    deterministic=False,
                )

            u_nom = out["u_nom"][0].cpu().numpy().astype(np.float32)
            u_safe = out["u_safe"][0].cpu().numpy().astype(np.float32)
            logp_agent = out["logp"][0].cpu().numpy().astype(np.float32)
            value_joint = float(out["value_joint"][0].cpu().numpy())
            slack_aux = float(out["slack_aux"][0].cpu().numpy()) if args.diff_qp else 0.0
            qp_fail_count = int(out.get("qp_fail_count", torch.zeros((1,), dtype=torch.int32))[0].cpu().item())
            qp_fail_ratio = float(qp_fail_count) / float(args.num_drones)

            action = env.velocity_to_normalized_action(u_safe)
            next_obs, env_reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

            desired_next = np.asarray(next_info["desired_positions"], dtype=np.float32)
            pos_now = np.asarray(obs[:, 0:3], dtype=np.float32)
            pos_next = np.asarray(next_obs[:, 0:3], dtype=np.float32)
            vel_now = np.asarray(obs[:, 3:6], dtype=np.float32)
            vel_next = np.asarray(next_obs[:, 3:6], dtype=np.float32)
            target_next = build_skill_targets(
                desired_next, current_skills, args.delta_x, args.delta_y, args.delta_z
            )
            err_now = np.linalg.norm(pos_now - target_now, axis=1)
            err_next = np.linalg.norm(pos_next - target_next, axis=1)
            skill_progress = float(np.mean(np.clip(err_now - err_next, -0.2, 0.2)))
            mean_err_next = float(np.mean(err_next))
            align_reward_raw = float(
                skill_velocity_alignment_reward(
                    u_safe,
                    current_skills,
                    cruise_speed=args.skill_cruise_speed,
                    accelerate_speed=args.skill_accelerate_speed,
                    decelerate_speed=args.skill_decelerate_speed,
                )
            )
            if args.align_distance_gating:
                gate = (mean_err_next - float(args.align_gate_near)) / max(
                    float(args.align_gate_far - args.align_gate_near), 1e-6
                )
                gate = float(np.clip(gate, 0.0, 1.0))
                gate = float(args.align_gate_min) + (1.0 - float(args.align_gate_min)) * gate
            else:
                gate = 1.0
            align_reward = gate * align_reward_raw
            skill_score = align_reward

            v_next_norm = np.linalg.norm(vel_next, axis=1)
            ref_dir, ref_speed = build_skill_reference_velocity(
                current_skills,
                cruise_speed=args.skill_cruise_speed,
                accelerate_speed=args.skill_accelerate_speed,
                decelerate_speed=args.skill_decelerate_speed,
                vel_now_n3=vel_now,
                speed_cap=env.cfg.max_target_speed_xy,
            )

            # Reward from user-provided formula:
            # r_L^i = -c1||a^i||^2 - c2((v^i-v_des)/v_des)^2 - c3((psi^i-psi_des)/pi)^2 - c4||d^i||^2
            action_pen = float(np.mean(np.sum(u_safe**2, axis=1)))
            vel_rel = (v_next_norm - ref_speed) / np.maximum(ref_speed, float(args.v_des_eps))
            vel_rel_pen = float(np.mean(vel_rel**2))

            psi = np.arctan2(vel_next[:, 1], vel_next[:, 0])
            psi_des = np.arctan2(ref_dir[:, 1], ref_dir[:, 0])
            dpsi = np.arctan2(np.sin(psi - psi_des), np.cos(psi - psi_des))
            heading_mask = (
                np.linalg.norm(ref_dir[:, 0:2], axis=1) > 1e-6
            ) & (np.linalg.norm(vel_next[:, 0:2], axis=1) > float(args.heading_speed_min))
            if np.any(heading_mask):
                heading_pen = float(np.mean((dpsi[heading_mask] / np.pi) ** 2))
            else:
                heading_pen = 0.0

            pos_sq_pen = float(np.mean(np.sum((pos_next - target_next) ** 2, axis=1)))
            intrinsic_reward = -(
                float(args.c1_action) * action_pen
                + float(args.c2_vel) * vel_rel_pen
                + float(args.c3_heading) * heading_pen
                + float(args.c4_pos) * pos_sq_pen
            )

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
                + intrinsic_reward
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
            obs_cbf_A_buf[t] = obs_A_now
            obs_cbf_b_buf[t] = obs_b_now
            target_buf[t] = target_now
            u_nom_buf[t] = u_nom
            logp_buf[t] = logp_agent
            val_buf[t] = value_joint
            rew_buf[t] = reward
            done_buf[t] = float(done)
            skill_score_buf[t] = skill_score
            align_raw_buf[t] = align_reward_raw
            align_gate_buf[t] = gate
            align_buf[t] = align_reward
            intrinsic_buf[t] = intrinsic_reward
            action_pen_buf[t] = action_pen
            vel_rel_pen_buf[t] = vel_rel_pen
            heading_pen_buf[t] = heading_pen
            pos_sq_pen_buf[t] = pos_sq_pen
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
            age_after = skill_ages + 1

            # Phi_z(o,t)=1 if o in J_z or timeout; otherwise 0.
            term_mask, succ_mask, tout_mask = evaluate_skill_termination(
                pos_n3=pos_next,
                vel_n3=vel_next,
                target_n3=target_next,
                skill_idx_n=current_skills,
                age_after_n=age_after,
                t_max_n=skill_periods,
                ref_dir_n3=ref_dir,
                ref_speed_n=ref_speed,
                pos_tol=args.skill_term_pos_tol,
                speed_tol=args.skill_term_speed_tol,
                heading_tol=args.skill_term_heading_tol,
                min_duration=args.skill_min_duration,
                pos_guard_scale=args.skill_term_pos_guard_scale,
            )
            term_success_buf[t] = float(np.mean(succ_mask.astype(np.float32)))
            term_timeout_buf[t] = float(np.mean(tout_mask.astype(np.float32)))
            skill_ages = age_after

            if not args.lock_skill_per_episode:
                for i in range(args.num_drones):
                    if not term_mask[i]:
                        continue
                    if np.random.rand() > args.skill_stay_prob:
                        candidate = int(np.random.choice(allowed_skills))
                    else:
                        candidate = int(current_skills[i])
                    if not in_skill_initiation_set(obs[i], min_altitude=args.skill_init_min_alt):
                        candidate = 0
                    current_skills[i] = candidate
                    skill_periods[i] = int(np.random.randint(args.skill_period_min, args.skill_period_max + 1))
                    skill_ages[i] = 0
                    switched_mask_next[i] = True
                    switch_count += 1

            # Bootstrap value from the true post-step state before any env.reset().
            with torch.no_grad():
                tau_next = np.clip(skill_ages / np.maximum(skill_periods - 1, 1), 0.0, 1.0).astype(np.float32)
                desired_next_for_value = np.asarray(info["desired_positions"], dtype=np.float32)
                target_next_for_value = build_skill_targets(
                    desired_next_for_value, current_skills, args.delta_x, args.delta_y, args.delta_z
                )
                policy_obs_next = build_policy_obs(
                    obs,
                    target_next_for_value,
                    use_delta_feature=args.use_delta_feature,
                    normalize_obs=args.normalize_obs,
                    num_drones=args.num_drones,
                    max_sensed_obstacles=env.cfg.max_sensed_obstacles,
                    include_neighbor_features=env.cfg.include_neighbor_features,
                    pos_scale=args.obs_pos_scale,
                    vel_scale=args.obs_vel_scale,
                    clip_val=args.obs_clip,
                )
                obs_next_t = torch.as_tensor(policy_obs_next[None, ...], dtype=torch.float32, device=device)
                skill_next_t = torch.as_tensor(current_skills[None, ...], dtype=torch.long, device=device)
                tau_next_t = torch.as_tensor(tau_next[None, :, None], dtype=torch.float32, device=device)
                _, _, value_next_agent = model.forward(obs_next_t, skill_next_t, tau_next_t)
                next_val_buf[t] = float(value_next_agent.mean(dim=-1).cpu().numpy()[0])

            # True terminal includes env termination + manual debug termination; pure timeout is non-terminal.
            terminal_buf[t] = float(bool(terminated) or bool(done and not truncated))

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
                for i in range(args.num_drones):
                    if not in_skill_initiation_set(obs[i], min_altitude=args.skill_init_min_alt):
                        current_skills[i] = 0
                switched_mask_next[:] = False
                desired_reset = np.asarray(info["desired_positions"], dtype=np.float32)
                target_reset = build_skill_targets(
                    desired_reset, current_skills, args.delta_x, args.delta_y, args.delta_z
                )
                pos_reset = np.asarray(obs[:, 0:3], dtype=np.float32)
                err_reset = np.linalg.norm(pos_reset - target_reset, axis=1)
                last_progress_dist_n[:] = err_reset.astype(np.float32)

        adv, ret = compute_gae(
            rewards=rew_buf,
            values=val_buf,
            next_values=next_val_buf,
            episode_dones=done_buf,
            terminals=terminal_buf,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(obs_buf, dtype=torch.float32, device=device)
        skill_t = torch.as_tensor(skill_buf, dtype=torch.long, device=device)
        tau_t = torch.as_tensor(tau_buf, dtype=torch.float32, device=device)
        cbf_t = torch.as_tensor(cbf_buf, dtype=torch.float32, device=device)
        obs_cbf_A_t = torch.as_tensor(obs_cbf_A_buf, dtype=torch.float32, device=device)
        obs_cbf_b_t = torch.as_tensor(obs_cbf_b_buf, dtype=torch.float32, device=device)
        target_t = torch.as_tensor(target_buf, dtype=torch.float32, device=device)
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
                b_obsA = obs_cbf_A_t[idx]
                b_obsb = obs_cbf_b_t[idx]
                b_target = target_t[idx]
                b_nom = u_nom_t[idx]
                b_old_logp = old_logp_t[idx]
                b_ret = ret_t[idx]
                b_adv = adv_t[idx]

                mean_b, std_b, value_agent_b = model.forward(b_obs, b_skill, b_tau)
                value_b = value_agent_b.mean(dim=-1)
                if args.low_level_qp_only:
                    policy_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                    entropy_b = torch.tensor(0.0, dtype=torch.float32, device=device)
                else:
                    dist_b = Normal(mean_b, std_b)
                    # Per-agent PPO ratio (MAPPO-style), avoid high-dimensional joint log-prob explosion.
                    logp_b = dist_b.log_prob(b_nom).sum(dim=-1)  # (B, N)
                    entropy_b = dist_b.entropy().sum(dim=-1).mean()
                qp_h_b, qp_f_b = model.qp_objective_params(
                    b_obs,
                    b_skill,
                    b_tau,
                    None if args.low_level_qp_only else mean_b,
                    nominal_anchor_weight=u_nom_anchor_w,
                )

                if not args.low_level_qp_only:
                    b_adv_expanded = b_adv.unsqueeze(1).expand(-1, args.num_drones)
                    ratio = torch.exp(logp_b - b_old_logp)  # (B, N)
                    surr1 = ratio * b_adv_expanded
                    surr2 = torch.clamp(ratio, 1.0 - ppo_cfg.clip_ratio, 1.0 + ppo_cfg.clip_ratio) * b_adv_expanded
                    policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((value_b - b_ret) ** 2).mean()

                qp_aux = torch.tensor(0.0, dtype=torch.float32, device=device)
                if args.diff_qp and qp_solver is not None:
                    safe_list = []
                    slack_list = []
                    for j in range(mean_b.shape[0]):
                        try:
                            qp_v_des_j = torch.zeros_like(mean_b[j]) if args.low_level_qp_only else mean_b[j]
                            u_safe_j, slack_j = qp_solver.solve_torch(
                                b_cbf[j],
                                qp_v_des_j,
                                target_pos_t=b_target[j],
                                target_vel_t=torch.zeros_like(b_target[j]),
                                skill_idx_t=b_skill[j],
                                obstacle_A_t=b_obsA[j],
                                obstacle_b_t=b_obsb[j],
                                h_diag_t=qp_h_b[j],
                                f_t=qp_f_b[j],
                            )
                        except Exception:
                            u_safe_j = torch.zeros_like(mean_b[j])
                            slack_j = torch.zeros((1,), dtype=mean_b.dtype, device=mean_b.device)
                        safe_list.append(u_safe_j)
                        slack_list.append(torch.mean(slack_j**2))
                    safe_stack = torch.stack(safe_list, dim=0)
                    slack_stack = torch.stack(slack_list, dim=0)
                    if args.low_level_qp_only:
                        qp_aux = 0.1 * torch.mean(slack_stack)
                    else:
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
            f"rollout_align_raw={float(np.mean(align_raw_buf)):.3f} "
            f"rollout_align_gate={float(np.mean(align_gate_buf)):.3f} "
            f"rollout_align={float(np.mean(align_buf)):.3f} "
            f"rollout_intrinsic={float(np.mean(intrinsic_buf)):.4f} "
            f"rollout_action_pen={float(np.mean(action_pen_buf)):.4f} "
            f"rollout_vel_rel_pen={float(np.mean(vel_rel_pen_buf)):.4f} "
            f"rollout_heading_pen={float(np.mean(heading_pen_buf)):.4f} "
            f"rollout_pos_sq_pen={float(np.mean(pos_sq_pen_buf)):.4f} "
            f"rollout_skill_progress={float(np.mean(skill_prog_buf)):.4f} "
            f"rollout_skill_err={float(np.mean(skill_err_buf)):.3f} "
            f"rollout_term_succ={float(np.mean(term_success_buf)):.3f} "
            f"rollout_term_timeout={float(np.mean(term_timeout_buf)):.3f} "
            f"rollout_intervene={float(np.mean(intervene_buf)):.4f} "
            f"rollout_slack={float(np.mean(slack_buf)):.4f} "
            f"rollout_qp_fail={float(np.mean(qp_fail_buf)):.4f} "
            f"rollout_contact={float(np.mean(contact_buf)):.4f} "
            f"switches={switch_count} qp_aux={last_qp_aux:.4f} diff_qp={args.diff_qp} "
            f"ll_qp_only={args.low_level_qp_only} "
            f"u_nom_anchor_w={u_nom_anchor_w:.3f} "
            f"debug_mode={args.debug_progress_reward} "
            f"debug_dist={float(np.mean(debug_dist_buf)):.3f} "
            f"debug_reward={float(np.mean(debug_reward_buf)):.3f}"
        )

        if args.save_path and (update % max(int(args.save_every), 1) == 0 or update == args.total_updates):
            save_dir = os.path.dirname(args.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update,
                "args": vars(args),
                "obs_dim": int(obs_dim),
            }
            torch.save(payload, args.save_path)

    env.close()


if __name__ == "__main__":
    main()
