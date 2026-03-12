import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from drone_env import (  # noqa: E402
    DifferentiableCBFQPConfig,
    HighLevelPolicyConfig,
    JointSkillActorCritic,
    LocalObstacleEnvConfig,
    LocalObstacleFormationEnv,
    SkillConditionedActorCritic,
    SkillConditionedPolicyConfig,
    build_solver_from_velocity_bounds,
)
from train_low_level_skill_ppo import (  # noqa: E402
    build_policy_obs,
    build_skill_reference_velocity,
    build_skill_targets,
    evaluate_skill_termination,
    in_skill_initiation_set,
    linear_anneal_weight,
)


@dataclass
class PPOConfig:
    rollout_steps: int
    epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip_ratio: float
    value_coef: float
    entropy_coef: float
    lr: float
    max_grad_norm: float


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


def parse_skill_set(skill_set_text: str, num_skills: int) -> np.ndarray:
    out: List[int] = []
    for token in skill_set_text.split(","):
        t = token.strip()
        if not t:
            continue
        v = int(t)
        if v < 0 or v >= num_skills:
            raise ValueError(f"Skill id {v} out of range [0, {num_skills - 1}]")
        out.append(v)
    if not out:
        raise ValueError("skill-set cannot be empty")
    return np.asarray(sorted(set(out)), dtype=np.int64)


def pairwise_distances(pos: np.ndarray) -> np.ndarray:
    n = pos.shape[0]
    out: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(np.linalg.norm(pos[i] - pos[j])))
    return np.asarray(out, dtype=np.float32)


def build_high_level_obs(info: dict, num_drones: int, max_sensed_obstacles: int, sensing_radius: float) -> np.ndarray:
    cbf_state = np.asarray(info.get("cbf_state", np.zeros((num_drones, 6), dtype=np.float32)), dtype=np.float32)
    if cbf_state.shape != (num_drones, 6):
        cbf_state = np.zeros((num_drones, 6), dtype=np.float32)
    desired = np.asarray(info.get("desired_positions", cbf_state[:, 0:3]), dtype=np.float32)
    if desired.shape != (num_drones, 3):
        desired = cbf_state[:, 0:3]
    err = desired - cbf_state[:, 0:3]
    pairwise = pairwise_distances(cbf_state[:, 0:3])
    sensed_cnt = np.asarray(info.get("sensed_obstacle_count", np.zeros((num_drones,), dtype=np.int32)), dtype=np.float32).reshape(-1)
    if sensed_cnt.size != num_drones:
        sensed_cnt = np.zeros((num_drones,), dtype=np.float32)
    sensed_cnt = sensed_cnt / max(float(max_sensed_obstacles), 1.0)
    min_clear = float(info.get("min_obstacle_clearance", float("inf")))
    clear_feat = 1.0 if not np.isfinite(min_clear) else float(np.clip(min_clear / max(float(sensing_radius), 1e-6), -2.0, 2.0))
    return np.concatenate(
        [
            cbf_state.reshape(-1),
            err.reshape(-1),
            pairwise.reshape(-1),
            sensed_cnt.reshape(-1),
            np.asarray([clear_feat], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def load_low_level_checkpoint(model: SkillConditionedActorCritic, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = None
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "low_level_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
        if state_dict is None and all(torch.is_tensor(v) for v in ckpt.values()):
            state_dict = ckpt
    if state_dict is None:
        raise RuntimeError(f"Unsupported low-level checkpoint format: {ckpt_path}")
    model.load_state_dict(state_dict, strict=True)


def build_skill_mask(num_skills: int, allowed_skills: np.ndarray, device: torch.device) -> torch.Tensor:
    mask = torch.full((num_skills,), -1e9, dtype=torch.float32, device=device)
    for s in allowed_skills.tolist():
        mask[int(s)] = 0.0
    return mask


def sample_periods(num_drones: int, k_min: int, k_max: int) -> np.ndarray:
    return np.random.randint(k_min, k_max + 1, size=(num_drones,), dtype=np.int64)


def intrinsic_reward(
    u_safe: np.ndarray,
    vel_now: np.ndarray,
    vel_next: np.ndarray,
    target_next: np.ndarray,
    pos_next: np.ndarray,
    current_skills: np.ndarray,
    cruise_speed: float,
    accelerate_speed: float,
    decelerate_speed: float,
    speed_cap: float,
    c1_action: float,
    c2_vel: float,
    c3_heading: float,
    c4_pos: float,
    v_des_eps: float,
    heading_speed_min: float,
) -> float:
    v_next_norm = np.linalg.norm(vel_next, axis=1)
    ref_dir, ref_speed = build_skill_reference_velocity(
        current_skills,
        cruise_speed=cruise_speed,
        accelerate_speed=accelerate_speed,
        decelerate_speed=decelerate_speed,
        vel_now_n3=vel_now,
        speed_cap=speed_cap,
    )
    action_pen = float(np.mean(np.sum(u_safe**2, axis=1)))
    vel_rel = (v_next_norm - ref_speed) / np.maximum(ref_speed, float(v_des_eps))
    vel_rel_pen = float(np.mean(vel_rel**2))

    psi = np.arctan2(vel_next[:, 1], vel_next[:, 0])
    psi_des = np.arctan2(ref_dir[:, 1], ref_dir[:, 0])
    dpsi = np.arctan2(np.sin(psi - psi_des), np.cos(psi - psi_des))
    heading_mask = (
        np.linalg.norm(ref_dir[:, 0:2], axis=1) > 1e-6
    ) & (np.linalg.norm(vel_next[:, 0:2], axis=1) > float(heading_speed_min))
    if np.any(heading_mask):
        heading_pen = float(np.mean((dpsi[heading_mask] / np.pi) ** 2))
    else:
        heading_pen = 0.0

    pos_sq_pen = float(np.mean(np.sum((pos_next - target_next) ** 2, axis=1)))
    return -(
        float(c1_action) * action_pen
        + float(c2_vel) * vel_rel_pen
        + float(c3_heading) * heading_pen
        + float(c4_pos) * pos_sq_pen
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint high-level and low-level PPO")
    parser.add_argument("--num-drones", type=int, default=4)
    parser.add_argument("--scenario", type=str, default="single_pillar", choices=["none", "bridge", "tree", "bridge_tree", "single_pillar"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--episode-len-sec", type=float, default=20.0)
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

    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-skills", type=int, default=7)
    parser.add_argument("--skill-set", type=str, default="0,1,2,3,4,5,6")
    parser.add_argument("--skill-period-min", type=int, default=32)
    parser.add_argument("--skill-period-max", type=int, default=96)
    parser.add_argument("--skill-min-duration", type=int, default=8)
    parser.add_argument("--skill-term-pos-tol", type=float, default=0.18)
    parser.add_argument("--skill-term-speed-tol", type=float, default=0.18)
    parser.add_argument("--skill-term-heading-tol", type=float, default=0.40)
    parser.add_argument("--skill-term-pos-guard-scale", type=float, default=3.0)
    parser.add_argument("--skill-init-min-alt", type=float, default=0.2)
    parser.add_argument("--delta-x", type=float, default=0.35)
    parser.add_argument("--delta-y", type=float, default=0.35)
    parser.add_argument("--delta-z", type=float, default=0.35)
    parser.add_argument("--sensing-radius", type=float, default=3.0)
    parser.add_argument("--max-sensed-obstacles", type=int, default=5)
    parser.add_argument("--safe-distance", type=float, default=0.22)
    parser.add_argument("--fall-z-threshold", type=float, default=0.5)
    parser.add_argument("--fall-penalty", type=float, default=10.0)
    parser.add_argument("--terminate-on-fall", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--use-delta-feature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-pos-scale", type=float, default=5.0)
    parser.add_argument("--obs-vel-scale", type=float, default=2.0)
    parser.add_argument("--obs-clip", type=float, default=5.0)

    parser.add_argument("--joint-train-low-level", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-level-checkpoint", type=str, default="")
    parser.add_argument("--low-hidden-dim", type=int, default=256)
    parser.add_argument("--low-level-deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--low-level-qp-only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--ll-lr", type=float, default=3e-4)
    parser.add_argument("--ll-clip-ratio", type=float, default=0.2)
    parser.add_argument("--ll-entropy-coef", type=float, default=0.01)
    parser.add_argument("--ll-value-coef", type=float, default=0.5)
    parser.add_argument("--ll-max-grad-norm", type=float, default=0.8)
    parser.add_argument("--ll-gamma", type=float, default=0.99)
    parser.add_argument("--ll-gae-lambda", type=float, default=0.95)
    parser.add_argument("--ll-epochs", type=int, default=6)
    parser.add_argument("--ll-minibatch-size", type=int, default=64)
    parser.add_argument("--ll-qp-aux-weight", type=float, default=0.1)

    parser.add_argument("--hl-hidden-dim", type=int, default=256)
    parser.add_argument("--hl-lr", type=float, default=3e-4)
    parser.add_argument("--hl-clip-ratio", type=float, default=0.2)
    parser.add_argument("--hl-entropy-coef", type=float, default=0.01)
    parser.add_argument("--hl-value-coef", type=float, default=0.5)
    parser.add_argument("--hl-max-grad-norm", type=float, default=0.8)
    parser.add_argument("--hl-gamma", type=float, default=0.99)
    parser.add_argument("--hl-gae-lambda", type=float, default=0.95)
    parser.add_argument("--hl-epochs", type=int, default=6)
    parser.add_argument("--hl-minibatch-size", type=int, default=64)

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

    parser.add_argument("--c1-action", type=float, default=0.05)
    parser.add_argument("--c2-vel", type=float, default=0.60)
    parser.add_argument("--c3-heading", type=float, default=0.25)
    parser.add_argument("--c4-pos", type=float, default=0.35)
    parser.add_argument("--v-des-eps", type=float, default=0.10)
    parser.add_argument("--heading-speed-min", type=float, default=0.05)

    parser.add_argument("--contact-penalty", type=float, default=5.0)
    parser.add_argument("--clearance-penalty", type=float, default=2.0)
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    if args.skill_period_min <= 0 or args.skill_period_max < args.skill_period_min:
        raise SystemExit("Invalid skill period range")
    if args.u_nom_anchor_anneal_start_frac < 0.0 or args.u_nom_anchor_anneal_end_frac > 1.0:
        raise SystemExit("u-nom-anchor anneal fractions must be within [0,1]")
    if args.u_nom_anchor_anneal_end_frac < args.u_nom_anchor_anneal_start_frac:
        raise SystemExit("u-nom-anchor-anneal-end-frac must be >= start-frac")
    if args.fall_z_threshold < 0.0 or args.fall_penalty < 0.0:
        raise SystemExit("fall-z-threshold and fall-penalty must be non-negative")
    if args.goal_speed < 0.0:
        raise SystemExit("goal-speed must be non-negative")
    if args.skill_cruise_speed < 0.0 or args.skill_accelerate_speed < 0.0 or args.skill_decelerate_speed < 0.0:
        raise SystemExit("skill speeds must be non-negative")
    if args.low_level_checkpoint and not os.path.isfile(args.low_level_checkpoint):
        raise SystemExit(f"Low-level checkpoint not found: {args.low_level_checkpoint}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    allowed_skills = parse_skill_set(args.skill_set, args.num_skills)
    fallback_skill = int(allowed_skills[0])

    env = LocalObstacleFormationEnv(
        LocalObstacleEnvConfig(
            num_drones=args.num_drones,
            gui=args.gui,
            episode_len_sec=args.episode_len_sec,
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
            sensing_radius=args.sensing_radius,
            max_sensed_obstacles=args.max_sensed_obstacles,
            scenario=args.scenario,
            single_pillar_x=args.single_pillar_x,
            single_pillar_y=args.single_pillar_y,
            safe_distance=args.safe_distance,
            fall_z_threshold=args.fall_z_threshold,
            fall_penalty=args.fall_penalty,
            terminate_on_fall=args.terminate_on_fall,
        )
    )
    obs, info = env.reset(seed=args.seed)

    desired_init = np.asarray(info["desired_positions"], dtype=np.float32)
    target_init = build_skill_targets(desired_init, np.zeros((args.num_drones,), dtype=np.int64), args.delta_x, args.delta_y, args.delta_z)
    low_obs0 = build_policy_obs(
        obs,
        target_init,
        use_delta_feature=args.use_delta_feature,
        normalize_obs=args.normalize_obs,
        num_drones=args.num_drones,
        max_sensed_obstacles=args.max_sensed_obstacles,
        include_neighbor_features=env.cfg.include_neighbor_features,
        pos_scale=args.obs_pos_scale,
        vel_scale=args.obs_vel_scale,
        clip_val=args.obs_clip,
    )
    low_obs_dim = int(low_obs0.shape[1])
    high_obs_dim = int(build_high_level_obs(info, args.num_drones, args.max_sensed_obstacles, args.sensing_radius).shape[0])

    low_model = SkillConditionedActorCritic(
        obs_local_dim=low_obs_dim,
        cfg=SkillConditionedPolicyConfig(
            num_skills=args.num_skills,
            hidden_dim=args.low_hidden_dim,
            action_scale_xy=env.cfg.max_target_speed_xy,
            action_scale_z=env.cfg.max_target_speed_z,
            use_parametric_qp=args.use_parametric_qp_objective,
            qp_h_base=args.qp_h_base,
            qp_h_min=args.qp_h_min,
            qp_f_scale=args.qp_f_scale,
            qp_only=args.low_level_qp_only,
        ),
    ).to(device)
    if args.low_level_checkpoint:
        load_low_level_checkpoint(low_model, args.low_level_checkpoint, device)

    if args.joint_train_low_level:
        low_model.train()
        for p in low_model.parameters():
            p.requires_grad_(True)
        ll_optimizer = optim.Adam(low_model.parameters(), lr=args.ll_lr)
    else:
        low_model.eval()
        for p in low_model.parameters():
            p.requires_grad_(False)
        ll_optimizer = None

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
                enable_clf=args.qp_enable_clf,
                clf_mode=args.qp_clf_mode,
                clf_rate=args.qp_clf_rate,
                heading_clf_rate=args.qp_heading_clf_rate,
                clf_deadzone=args.qp_clf_deadzone,
                speed_clf_deadzone=args.qp_speed_clf_deadzone,
                heading_clf_deadzone=args.qp_heading_clf_deadzone,
                clf_slack_weight=args.qp_clf_slack_weight,
                cruise_speed=args.skill_cruise_speed,
                accelerate_speed=args.skill_accelerate_speed,
                decelerate_speed=args.skill_decelerate_speed,
                skill_speed_cap=env.cfg.max_target_speed_xy,
                qp_h_min=args.qp_h_min,
                max_obstacle_constraints_per_drone=args.max_sensed_obstacles,
            ),
            device=str(device),
        )

    high_model = JointSkillActorCritic(
        global_obs_dim=high_obs_dim,
        cfg=HighLevelPolicyConfig(num_drones=args.num_drones, num_skills=args.num_skills, hidden_dim=args.hl_hidden_dim),
    ).to(device)
    hl_optimizer = optim.Adam(high_model.parameters(), lr=args.hl_lr)

    hl_cfg = PPOConfig(args.rollout_steps, args.hl_epochs, args.hl_minibatch_size, args.hl_gamma, args.hl_gae_lambda, args.hl_clip_ratio, args.hl_value_coef, args.hl_entropy_coef, args.hl_lr, args.hl_max_grad_norm)
    ll_cfg = PPOConfig(args.rollout_steps, args.ll_epochs, args.ll_minibatch_size, args.ll_gamma, args.ll_gae_lambda, args.ll_clip_ratio, args.ll_value_coef, args.ll_entropy_coef, args.ll_lr, args.ll_max_grad_norm)
    skill_mask = build_skill_mask(args.num_skills, allowed_skills, device)

    current_skills = np.full((args.num_drones,), fallback_skill, dtype=np.int64)
    skill_periods = sample_periods(args.num_drones, args.skill_period_min, args.skill_period_max)
    skill_ages = np.zeros((args.num_drones,), dtype=np.int64)
    pending_update = np.ones((args.num_drones,), dtype=bool)
    recent_ext_returns: List[float] = []
    recent_int_returns: List[float] = []
    ep_ext = 0.0
    ep_int = 0.0

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
        hl_obs_buf = np.zeros((hl_cfg.rollout_steps, high_obs_dim), dtype=np.float32)
        hl_skill_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones), dtype=np.int64)
        hl_logp_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones), dtype=np.float32)
        hl_mask_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones), dtype=np.float32)
        hl_rew_buf = np.zeros((hl_cfg.rollout_steps,), dtype=np.float32)
        hl_val_buf = np.zeros((hl_cfg.rollout_steps,), dtype=np.float32)
        hl_done_buf = np.zeros((hl_cfg.rollout_steps,), dtype=np.float32)

        ll_obs_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones, low_obs_dim), dtype=np.float32)
        ll_skill_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones), dtype=np.int64)
        ll_tau_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones, 1), dtype=np.float32)
        ll_cbf_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones, 6), dtype=np.float32)
        ll_obsA_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones, args.max_sensed_obstacles, 3), dtype=np.float32)
        ll_obsb_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones, args.max_sensed_obstacles), dtype=np.float32)
        ll_target_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones, 3), dtype=np.float32)
        ll_nom_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones, 3), dtype=np.float32)
        ll_logp_buf = np.zeros((hl_cfg.rollout_steps, args.num_drones), dtype=np.float32)
        ll_val_buf = np.zeros((hl_cfg.rollout_steps,), dtype=np.float32)
        ll_rew_buf = np.zeros((hl_cfg.rollout_steps,), dtype=np.float32)
        ll_done_buf = np.zeros((hl_cfg.rollout_steps,), dtype=np.float32)

        term_succ_buf = np.zeros((hl_cfg.rollout_steps,), dtype=np.float32)
        term_timeout_buf = np.zeros((hl_cfg.rollout_steps,), dtype=np.float32)
        contact_buf = np.zeros((hl_cfg.rollout_steps,), dtype=np.float32)
        clear_buf = np.zeros((hl_cfg.rollout_steps,), dtype=np.float32)
        qp_fail_buf = np.zeros((hl_cfg.rollout_steps,), dtype=np.float32)
        switches = 0
        decision_count = 0.0

        ll_rollout_deterministic = (not args.joint_train_low_level) and bool(args.low_level_deterministic)
        high_model.train()
        if args.joint_train_low_level:
            low_model.train()
        else:
            low_model.eval()

        for t in range(hl_cfg.rollout_steps):
            gobs = build_high_level_obs(info, args.num_drones, args.max_sensed_obstacles, args.sensing_radius)
            hl_obs_buf[t] = gobs
            gobs_t = torch.as_tensor(gobs[None, :], dtype=torch.float32, device=device)
            decision_mask = pending_update.astype(np.float32)
            hl_mask_buf[t] = decision_mask
            decision_count += float(np.sum(decision_mask))

            with torch.no_grad():
                logits_t, value_t = high_model.forward(gobs_t)
                logits_t = logits_t + skill_mask.view(1, 1, -1)
                dist_t = Categorical(logits=logits_t)
                sampled_t = dist_t.sample()
                sampled_np = sampled_t[0].cpu().numpy().astype(np.int64)
                logp_np = dist_t.log_prob(sampled_t)[0].cpu().numpy().astype(np.float32)

            hl_val_buf[t] = float(value_t[0].cpu().numpy())
            hl_logp_buf[t] = 0.0
            if np.any(pending_update):
                cbf_now = np.asarray(info.get("cbf_state", np.zeros((args.num_drones, 6), dtype=np.float32)), dtype=np.float32)
                for i in range(args.num_drones):
                    if not pending_update[i]:
                        continue
                    candidate = int(sampled_np[i])
                    if not in_skill_initiation_set(cbf_now[i], min_altitude=args.skill_init_min_alt):
                        candidate = fallback_skill
                    current_skills[i] = candidate
                    skill_periods[i] = int(np.random.randint(args.skill_period_min, args.skill_period_max + 1))
                    skill_ages[i] = 0
                    hl_logp_buf[t, i] = float(logp_np[i])
                    switches += 1
            hl_skill_buf[t] = current_skills

            desired_now = np.asarray(info["desired_positions"], dtype=np.float32)
            target_now = build_skill_targets(desired_now, current_skills, args.delta_x, args.delta_y, args.delta_z)
            low_obs = build_policy_obs(
                obs,
                target_now,
                use_delta_feature=args.use_delta_feature,
                normalize_obs=args.normalize_obs,
                num_drones=args.num_drones,
                max_sensed_obstacles=args.max_sensed_obstacles,
                include_neighbor_features=env.cfg.include_neighbor_features,
                pos_scale=args.obs_pos_scale,
                vel_scale=args.obs_vel_scale,
                clip_val=args.obs_clip,
            )
            tau_vec = np.clip(skill_ages / np.maximum(skill_periods - 1, 1), 0.0, 1.0).astype(np.float32)
            cbf_state = np.asarray(info["cbf_state"], dtype=np.float32)
            obs_A_now = np.asarray(
                info.get("obstacle_cbf_A", np.zeros((args.num_drones, args.max_sensed_obstacles, 3), dtype=np.float32)),
                dtype=np.float32,
            )
            obs_b_now = np.asarray(
                info.get("obstacle_cbf_b", np.zeros((args.num_drones, args.max_sensed_obstacles), dtype=np.float32)),
                dtype=np.float32,
            )

            with torch.no_grad():
                ll_out = low_model.act(
                    obs_local=torch.as_tensor(low_obs[None, ...], dtype=torch.float32, device=device),
                    skill_idx=torch.as_tensor(current_skills[None, ...], dtype=torch.long, device=device),
                    tau=torch.as_tensor(tau_vec[None, :, None], dtype=torch.float32, device=device),
                    cbf_state=torch.as_tensor(cbf_state[None, ...], dtype=torch.float32, device=device),
                    qp_target_pos=torch.as_tensor(target_now[None, ...], dtype=torch.float32, device=device),
                    qp_target_vel=torch.zeros((1, args.num_drones, 3), dtype=torch.float32, device=device),
                    qp_skill_idx=torch.as_tensor(current_skills[None, ...], dtype=torch.long, device=device),
                    qp_obstacle_A=torch.as_tensor(obs_A_now[None, ...], dtype=torch.float32, device=device),
                    qp_obstacle_b=torch.as_tensor(obs_b_now[None, ...], dtype=torch.float32, device=device),
                    qp_solver=qp_solver,
                    qp_nominal_anchor_weight=u_nom_anchor_w,
                    deterministic=ll_rollout_deterministic,
                )
            u_nom = ll_out["u_nom"][0].cpu().numpy().astype(np.float32)
            u_safe = ll_out["u_safe"][0].cpu().numpy().astype(np.float32)
            ll_logp = ll_out["logp"][0].cpu().numpy().astype(np.float32)
            ll_value = float(ll_out["value_joint"][0].cpu().numpy())
            qp_fail_count = int(ll_out.get("qp_fail_count", torch.zeros((1,), dtype=torch.int32))[0].cpu().item())
            qp_fail_ratio = float(qp_fail_count) / float(args.num_drones)

            action = env.velocity_to_normalized_action(u_safe)
            next_obs, env_reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

            cbf_next = np.asarray(next_info["cbf_state"], dtype=np.float32)
            vel_now = np.asarray(cbf_state[:, 3:6], dtype=np.float32)
            pos_next = cbf_next[:, 0:3]
            vel_next = cbf_next[:, 3:6]
            desired_next = np.asarray(next_info["desired_positions"], dtype=np.float32)
            target_next = build_skill_targets(desired_next, current_skills, args.delta_x, args.delta_y, args.delta_z)

            int_r = intrinsic_reward(
                u_safe=u_safe,
                vel_now=vel_now,
                vel_next=vel_next,
                target_next=target_next,
                pos_next=pos_next,
                current_skills=current_skills,
                cruise_speed=args.skill_cruise_speed,
                accelerate_speed=args.skill_accelerate_speed,
                decelerate_speed=args.skill_decelerate_speed,
                speed_cap=env.cfg.max_target_speed_xy,
                c1_action=args.c1_action,
                c2_vel=args.c2_vel,
                c3_heading=args.c3_heading,
                c4_pos=args.c4_pos,
                v_des_eps=args.v_des_eps,
                heading_speed_min=args.heading_speed_min,
            )
            contact = float(int(next_info.get("obstacle_contact_count", 0) > 0))
            min_clear = float(next_info.get("min_obstacle_clearance", float("inf")))
            clear_vio = max(0.0, -min_clear) if np.isfinite(min_clear) else 0.0
            ext_r = float(env_reward) - float(args.contact_penalty) * contact - float(args.clearance_penalty) * clear_vio

            hl_rew_buf[t] = ext_r
            hl_done_buf[t] = float(done)
            ll_rew_buf[t] = int_r
            ll_done_buf[t] = float(done)
            ll_val_buf[t] = ll_value
            contact_buf[t] = contact
            clear_buf[t] = clear_vio
            qp_fail_buf[t] = qp_fail_ratio

            ll_obs_buf[t] = low_obs
            ll_skill_buf[t] = current_skills
            ll_tau_buf[t, :, 0] = tau_vec
            ll_cbf_buf[t] = cbf_state
            ll_obsA_buf[t] = obs_A_now
            ll_obsb_buf[t] = obs_b_now
            ll_target_buf[t] = target_now
            ll_nom_buf[t] = u_nom
            ll_logp_buf[t] = ll_logp

            ep_ext += ext_r
            ep_int += int_r

            ref_dir, ref_speed = build_skill_reference_velocity(
                current_skills,
                cruise_speed=args.skill_cruise_speed,
                accelerate_speed=args.skill_accelerate_speed,
                decelerate_speed=args.skill_decelerate_speed,
                vel_now_n3=vel_now,
                speed_cap=env.cfg.max_target_speed_xy,
            )
            age_after = skill_ages + 1
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
            term_succ_buf[t] = float(np.mean(succ_mask.astype(np.float32)))
            term_timeout_buf[t] = float(np.mean(tout_mask.astype(np.float32)))
            pending_update = term_mask.copy()
            skill_ages = age_after
            obs, info = next_obs, next_info

            if done:
                recent_ext_returns.append(ep_ext)
                recent_int_returns.append(ep_int)
                ep_ext = 0.0
                ep_int = 0.0
                obs, info = env.reset()
                current_skills[:] = fallback_skill
                skill_periods = sample_periods(args.num_drones, args.skill_period_min, args.skill_period_max)
                skill_ages[:] = 0
                pending_update[:] = True

        with torch.no_grad():
            hl_last_obs = build_high_level_obs(info, args.num_drones, args.max_sensed_obstacles, args.sensing_radius)
            _, hl_v_last = high_model.forward(torch.as_tensor(hl_last_obs[None, :], dtype=torch.float32, device=device))
        hl_adv, hl_ret = compute_gae(hl_rew_buf, hl_val_buf, hl_done_buf, float(hl_v_last[0].cpu().numpy()), hl_cfg.gamma, hl_cfg.gae_lambda)
        hl_adv = (hl_adv - hl_adv.mean()) / (hl_adv.std() + 1e-8)

        hl_obs_t = torch.as_tensor(hl_obs_buf, dtype=torch.float32, device=device)
        hl_skill_t = torch.as_tensor(hl_skill_buf, dtype=torch.long, device=device)
        hl_old_logp_t = torch.as_tensor(hl_logp_buf, dtype=torch.float32, device=device)
        hl_mask_t = torch.as_tensor(hl_mask_buf, dtype=torch.float32, device=device)
        hl_adv_t = torch.as_tensor(hl_adv, dtype=torch.float32, device=device)
        hl_ret_t = torch.as_tensor(hl_ret, dtype=torch.float32, device=device)

        for _ in range(hl_cfg.epochs):
            for idx in batch_indices(hl_cfg.rollout_steps, hl_cfg.minibatch_size):
                b_obs = hl_obs_t[idx]
                b_skill = hl_skill_t[idx]
                b_old_logp = hl_old_logp_t[idx]
                b_mask = hl_mask_t[idx]
                b_adv = hl_adv_t[idx]
                b_ret = hl_ret_t[idx]

                logits_b, value_b = high_model.forward(b_obs)
                logits_b = logits_b + skill_mask.view(1, 1, -1)
                dist_b = Categorical(logits=logits_b)
                logp_b = dist_b.log_prob(b_skill)
                b_adv_expand = b_adv.unsqueeze(1).expand(-1, args.num_drones)
                ratio = torch.exp(logp_b - b_old_logp)
                surr1 = ratio * b_adv_expand * b_mask
                surr2 = torch.clamp(ratio, 1.0 - hl_cfg.clip_ratio, 1.0 + hl_cfg.clip_ratio) * b_adv_expand * b_mask
                denom = torch.clamp(b_mask.sum(), min=1.0)
                policy_loss = -torch.min(surr1, surr2).sum() / denom
                entropy = (dist_b.entropy() * b_mask).sum() / denom
                value_loss = 0.5 * ((value_b - b_ret) ** 2).mean()
                hl_loss = policy_loss + hl_cfg.value_coef * value_loss - hl_cfg.entropy_coef * entropy

                hl_optimizer.zero_grad(set_to_none=True)
                hl_loss.backward()
                nn.utils.clip_grad_norm_(high_model.parameters(), hl_cfg.max_grad_norm)
                hl_optimizer.step()

        last_ll_qp_aux = 0.0
        if args.joint_train_low_level and ll_optimizer is not None:
            with torch.no_grad():
                desired_last = np.asarray(info["desired_positions"], dtype=np.float32)
                target_last = build_skill_targets(desired_last, current_skills, args.delta_x, args.delta_y, args.delta_z)
                ll_obs_last = build_policy_obs(
                    obs,
                    target_last,
                    use_delta_feature=args.use_delta_feature,
                    normalize_obs=args.normalize_obs,
                    num_drones=args.num_drones,
                    max_sensed_obstacles=args.max_sensed_obstacles,
                    include_neighbor_features=env.cfg.include_neighbor_features,
                    pos_scale=args.obs_pos_scale,
                    vel_scale=args.obs_vel_scale,
                    clip_val=args.obs_clip,
                )
                tau_last = np.clip(skill_ages / np.maximum(skill_periods - 1, 1), 0.0, 1.0).astype(np.float32)
                _, _, v_last_agent = low_model.forward(
                    torch.as_tensor(ll_obs_last[None, ...], dtype=torch.float32, device=device),
                    torch.as_tensor(current_skills[None, ...], dtype=torch.long, device=device),
                    torch.as_tensor(tau_last[None, :, None], dtype=torch.float32, device=device),
                )
                ll_last_v = float(v_last_agent.mean(dim=-1).cpu().numpy()[0])

            ll_adv, ll_ret = compute_gae(ll_rew_buf, ll_val_buf, ll_done_buf, ll_last_v, ll_cfg.gamma, ll_cfg.gae_lambda)
            ll_adv = (ll_adv - ll_adv.mean()) / (ll_adv.std() + 1e-8)

            ll_obs_t = torch.as_tensor(ll_obs_buf, dtype=torch.float32, device=device)
            ll_skill_t = torch.as_tensor(ll_skill_buf, dtype=torch.long, device=device)
            ll_tau_t = torch.as_tensor(ll_tau_buf, dtype=torch.float32, device=device)
            ll_cbf_t = torch.as_tensor(ll_cbf_buf, dtype=torch.float32, device=device)
            ll_obsA_t = torch.as_tensor(ll_obsA_buf, dtype=torch.float32, device=device)
            ll_obsb_t = torch.as_tensor(ll_obsb_buf, dtype=torch.float32, device=device)
            ll_target_t = torch.as_tensor(ll_target_buf, dtype=torch.float32, device=device)
            ll_nom_t = torch.as_tensor(ll_nom_buf, dtype=torch.float32, device=device)
            ll_old_logp_t = torch.as_tensor(ll_logp_buf, dtype=torch.float32, device=device)
            ll_adv_t = torch.as_tensor(ll_adv, dtype=torch.float32, device=device)
            ll_ret_t = torch.as_tensor(ll_ret, dtype=torch.float32, device=device)

            for _ in range(ll_cfg.epochs):
                for idx in batch_indices(ll_cfg.rollout_steps, ll_cfg.minibatch_size):
                    b_obs = ll_obs_t[idx]
                    b_skill = ll_skill_t[idx]
                    b_tau = ll_tau_t[idx]
                    b_cbf = ll_cbf_t[idx]
                    b_obsA = ll_obsA_t[idx]
                    b_obsb = ll_obsb_t[idx]
                    b_target = ll_target_t[idx]
                    b_nom = ll_nom_t[idx]
                    b_old_logp = ll_old_logp_t[idx]
                    b_adv = ll_adv_t[idx]
                    b_ret = ll_ret_t[idx]

                    mean_b, std_b, value_agent_b = low_model.forward(b_obs, b_skill, b_tau)
                    value_b = value_agent_b.mean(dim=-1)
                    if args.low_level_qp_only:
                        ll_policy_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                        entropy_b = torch.tensor(0.0, dtype=torch.float32, device=device)
                    else:
                        dist_b = Normal(mean_b, std_b)
                        logp_b = dist_b.log_prob(b_nom).sum(dim=-1)
                        entropy_b = dist_b.entropy().sum(dim=-1).mean()
                        b_adv_expand = b_adv.unsqueeze(1).expand(-1, args.num_drones)
                        ratio = torch.exp(logp_b - b_old_logp)
                        surr1 = ratio * b_adv_expand
                        surr2 = torch.clamp(ratio, 1.0 - ll_cfg.clip_ratio, 1.0 + ll_cfg.clip_ratio) * b_adv_expand
                        ll_policy_loss = -torch.min(surr1, surr2).mean()
                    ll_value_loss = 0.5 * ((value_b - b_ret) ** 2).mean()

                    qp_aux = torch.tensor(0.0, dtype=torch.float32, device=device)
                    if args.diff_qp and qp_solver is not None and args.ll_qp_aux_weight > 0.0:
                        qp_u_nom_b = None if args.low_level_qp_only else mean_b
                        qp_h_b, qp_f_b = low_model.qp_objective_params(
                            b_obs,
                            b_skill,
                            b_tau,
                            qp_u_nom_b,
                            nominal_anchor_weight=u_nom_anchor_w,
                        )
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
                                slack_j = torch.zeros((args.num_drones,), dtype=mean_b.dtype, device=mean_b.device)
                            safe_list.append(u_safe_j)
                            slack_list.append(torch.mean(slack_j**2))
                        safe_stack = torch.stack(safe_list, dim=0)
                        slack_stack = torch.stack(slack_list, dim=0)
                        if args.low_level_qp_only:
                            qp_aux = 0.1 * torch.mean(slack_stack)
                        else:
                            qp_aux = torch.mean((safe_stack - mean_b) ** 2) + 0.1 * torch.mean(slack_stack)

                    ll_loss = ll_policy_loss + ll_cfg.value_coef * ll_value_loss - ll_cfg.entropy_coef * entropy_b + float(args.ll_qp_aux_weight) * qp_aux
                    ll_optimizer.zero_grad(set_to_none=True)
                    ll_loss.backward()
                    nn.utils.clip_grad_norm_(low_model.parameters(), ll_cfg.max_grad_norm)
                    ll_optimizer.step()
                    last_ll_qp_aux = float(qp_aux.detach().cpu().item())

        avg_ext = float(np.mean(recent_ext_returns[-10:])) if recent_ext_returns else 0.0
        avg_int = float(np.mean(recent_int_returns[-10:])) if recent_int_returns else 0.0
        decision_rate = float(decision_count / float(args.num_drones * hl_cfg.rollout_steps))
        print(
            f"update={update:04d} avg_ext_return_10={avg_ext:.3f} avg_int_return_10={avg_int:.3f} "
            f"rollout_ext={float(np.mean(hl_rew_buf)):.3f} rollout_int={float(np.mean(ll_rew_buf)):.3f} "
            f"decision_rate={decision_rate:.3f} switches={switches} "
            f"rollout_term_succ={float(np.mean(term_succ_buf)):.3f} rollout_term_timeout={float(np.mean(term_timeout_buf)):.3f} "
            f"rollout_contact={float(np.mean(contact_buf)):.4f} rollout_clear_vio={float(np.mean(clear_buf)):.4f} "
            f"rollout_qp_fail={float(np.mean(qp_fail_buf)):.4f} joint_ll={args.joint_train_low_level} ll_qp_only={args.low_level_qp_only} "
            f"u_nom_anchor_w={u_nom_anchor_w:.3f} ll_qp_aux={last_ll_qp_aux:.4f}"
        )

        if args.save_path and (update % max(int(args.save_every), 1) == 0 or update == args.total_updates):
            save_dir = os.path.dirname(args.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    "update": update,
                    "args": vars(args),
                    "high_model_state_dict": high_model.state_dict(),
                    "high_optimizer_state_dict": hl_optimizer.state_dict(),
                    "low_model_state_dict": low_model.state_dict(),
                    "low_optimizer_state_dict": None if ll_optimizer is None else ll_optimizer.state_dict(),
                    "obs_dim_high": high_obs_dim,
                    "obs_dim_low": low_obs_dim,
                },
                args.save_path,
            )

    env.close()


if __name__ == "__main__":
    main()
