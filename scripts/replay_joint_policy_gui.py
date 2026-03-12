import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

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
from train_high_level_policy_ppo import build_high_level_obs, build_skill_mask, parse_skill_set  # noqa: E402
from train_low_level_skill_ppo import (  # noqa: E402
    build_policy_obs,
    build_skill_reference_velocity,
    build_skill_targets,
    evaluate_skill_termination,
    in_skill_initiation_set,
    linear_anneal_weight,
)


def _ckpt_get(d: Dict, key: str, default):
    return d[key] if key in d else default


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay joint high-level/low-level checkpoint in GUI")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sleep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic-high", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic-low", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-gui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    if not isinstance(payload, dict):
        raise SystemExit("Unsupported checkpoint format: expected dict payload.")

    high_sd = payload.get("high_model_state_dict", None)
    low_sd = payload.get("low_model_state_dict", None)
    saved_args = payload.get("args", {})
    if high_sd is None or low_sd is None:
        raise SystemExit("Checkpoint is not a joint checkpoint (missing high/low model state).")

    num_drones = int(_ckpt_get(saved_args, "num_drones", 4))
    num_skills = int(_ckpt_get(saved_args, "num_skills", 7))
    skill_set = str(_ckpt_get(saved_args, "skill_set", "0,1,2,3,4,5,6"))
    allowed_skills = parse_skill_set(skill_set, num_skills)
    fallback_skill = int(allowed_skills[0])
    low_level_qp_only = bool(_ckpt_get(saved_args, "low_level_qp_only", False))
    max_sensed = int(_ckpt_get(saved_args, "max_sensed_obstacles", 3))
    sensing_radius = float(_ckpt_get(saved_args, "sensing_radius", 3.0))
    saved_total_updates = int(_ckpt_get(saved_args, "total_updates", max(int(payload.get("update", 1)), 1)))
    saved_update = int(payload.get("update", saved_total_updates))
    u_nom_anchor_w = linear_anneal_weight(
        update=saved_update,
        total_updates=saved_total_updates,
        start_weight=float(_ckpt_get(saved_args, "u_nom_anchor_start", 1.0)),
        end_weight=float(_ckpt_get(saved_args, "u_nom_anchor_end", 0.0)),
        start_frac=float(_ckpt_get(saved_args, "u_nom_anchor_anneal_start_frac", 0.0)),
        end_frac=float(_ckpt_get(saved_args, "u_nom_anchor_anneal_end_frac", 1.0)),
    )
    if low_level_qp_only:
        u_nom_anchor_w = 0.0

    env = LocalObstacleFormationEnv(
        LocalObstacleEnvConfig(
            num_drones=num_drones,
            gui=bool(args.gui),
            episode_len_sec=float(_ckpt_get(saved_args, "episode_len_sec", 20.0)),
            formation_pattern=str(_ckpt_get(saved_args, "formation_pattern", "line")),
            formation_spacing=float(_ckpt_get(saved_args, "formation_spacing", 0.35)),
            use_moving_goal=bool(_ckpt_get(saved_args, "use_moving_goal", False)),
            goal_start_x=float(_ckpt_get(saved_args, "goal_start_x", 0.0)),
            goal_start_y=float(_ckpt_get(saved_args, "goal_start_y", 0.0)),
            goal_start_z=float(_ckpt_get(saved_args, "goal_start_z", 1.0)),
            goal_end_x=float(_ckpt_get(saved_args, "goal_end_x", 0.0)),
            goal_end_y=float(_ckpt_get(saved_args, "goal_end_y", 0.0)),
            goal_end_z=float(_ckpt_get(saved_args, "goal_end_z", 1.0)),
            goal_speed=float(_ckpt_get(saved_args, "goal_speed", 0.5)),
            sensing_radius=sensing_radius,
            max_sensed_obstacles=max_sensed,
            scenario=str(_ckpt_get(saved_args, "scenario", "bridge_tree")),
            single_pillar_x=float(_ckpt_get(saved_args, "single_pillar_x", 2.0)),
            single_pillar_y=float(_ckpt_get(saved_args, "single_pillar_y", 0.0)),
            safe_distance=float(_ckpt_get(saved_args, "safe_distance", 0.22)),
            fall_z_threshold=float(_ckpt_get(saved_args, "fall_z_threshold", 0.15)),
            fall_penalty=float(_ckpt_get(saved_args, "fall_penalty", 10.0)),
            terminate_on_fall=bool(_ckpt_get(saved_args, "terminate_on_fall", True)),
        )
    )

    obs, info = env.reset(seed=args.seed)
    desired_init = np.asarray(info["desired_positions"], dtype=np.float32)
    target_init = build_skill_targets(
        desired_init,
        np.zeros((num_drones,), dtype=np.int64),
        float(_ckpt_get(saved_args, "delta_x", 0.35)),
        float(_ckpt_get(saved_args, "delta_y", 0.35)),
        float(_ckpt_get(saved_args, "delta_z", 0.35)),
    )
    low_obs0 = build_policy_obs(
        obs,
        target_init,
        use_delta_feature=bool(_ckpt_get(saved_args, "use_delta_feature", True)),
        normalize_obs=bool(_ckpt_get(saved_args, "normalize_obs", True)),
        num_drones=num_drones,
        max_sensed_obstacles=max_sensed,
        include_neighbor_features=env.cfg.include_neighbor_features,
        pos_scale=float(_ckpt_get(saved_args, "obs_pos_scale", 5.0)),
        vel_scale=float(_ckpt_get(saved_args, "obs_vel_scale", 2.0)),
        clip_val=float(_ckpt_get(saved_args, "obs_clip", 5.0)),
    )
    low_obs_dim = int(low_obs0.shape[1])
    high_obs_dim = int(build_high_level_obs(info, num_drones, max_sensed, sensing_radius).shape[0])

    high_model = JointSkillActorCritic(
        global_obs_dim=high_obs_dim,
        cfg=HighLevelPolicyConfig(
            num_drones=num_drones,
            num_skills=num_skills,
            hidden_dim=int(_ckpt_get(saved_args, "hl_hidden_dim", 256)),
        ),
    ).to(device)
    high_model.load_state_dict(high_sd, strict=True)
    high_model.eval()

    low_model = SkillConditionedActorCritic(
        obs_local_dim=low_obs_dim,
        cfg=SkillConditionedPolicyConfig(
            num_skills=num_skills,
            hidden_dim=int(_ckpt_get(saved_args, "low_hidden_dim", 256)),
            action_scale_xy=env.cfg.max_target_speed_xy,
            action_scale_z=env.cfg.max_target_speed_z,
            use_parametric_qp=bool(_ckpt_get(saved_args, "use_parametric_qp_objective", True)),
            qp_h_base=float(_ckpt_get(saved_args, "qp_h_base", 2.0)),
            qp_h_min=float(_ckpt_get(saved_args, "qp_h_min", 1e-3)),
            qp_f_scale=float(_ckpt_get(saved_args, "qp_f_scale", 0.5)),
            qp_only=low_level_qp_only,
        ),
    ).to(device)
    low_model.load_state_dict(low_sd, strict=True)
    print(
        f"[replay] using ll_qp_only={low_level_qp_only} "
        f"u_nom_anchor_w={u_nom_anchor_w:.3f} from checkpoint update={saved_update}"
    )
    low_model.eval()

    qp_solver = None
    if bool(_ckpt_get(saved_args, "diff_qp", True)):
        qp_solver = build_solver_from_velocity_bounds(
            num_drones=num_drones,
            max_speed_xy=env.cfg.max_target_speed_xy,
            max_speed_z=env.cfg.max_target_speed_z,
            config=DifferentiableCBFQPConfig(
                safe_distance=env.cfg.safe_distance,
                alpha=4.0,
                solve_method="SCS",
                enable_clf=bool(_ckpt_get(saved_args, "qp_enable_clf", True)),
                clf_mode=str(_ckpt_get(saved_args, "qp_clf_mode", "skill")),
                clf_rate=float(_ckpt_get(saved_args, "qp_clf_rate", 1.0)),
                heading_clf_rate=float(_ckpt_get(saved_args, "qp_heading_clf_rate", 2.0)),
                clf_deadzone=float(_ckpt_get(saved_args, "qp_clf_deadzone", 0.05)),
                speed_clf_deadzone=float(_ckpt_get(saved_args, "qp_speed_clf_deadzone", 0.08)),
                heading_clf_deadzone=float(_ckpt_get(saved_args, "qp_heading_clf_deadzone", 0.08)),
                clf_slack_weight=float(_ckpt_get(saved_args, "qp_clf_slack_weight", 80.0)),
                cruise_speed=float(_ckpt_get(saved_args, "skill_cruise_speed", 0.5)),
                accelerate_speed=float(_ckpt_get(saved_args, "skill_accelerate_speed", 0.2)),
                decelerate_speed=float(_ckpt_get(saved_args, "skill_decelerate_speed", 0.2)),
                skill_speed_cap=env.cfg.max_target_speed_xy,
                qp_h_min=float(_ckpt_get(saved_args, "qp_h_min", 1e-3)),
                max_obstacle_constraints_per_drone=max_sensed,
            ),
            device=str(device),
        )

    skill_mask = build_skill_mask(num_skills, allowed_skills, device)
    ctrl_dt = 1.0 / float(env.cfg.ctrl_freq)
    skill_period_min = int(_ckpt_get(saved_args, "skill_period_min", 32))
    skill_period_max = int(_ckpt_get(saved_args, "skill_period_max", 96))

    try:
        for ep in range(args.episodes):
            if ep > 0:
                obs, info = env.reset(seed=args.seed + ep)

            current_skills = np.full((num_drones,), fallback_skill, dtype=np.int64)
            skill_periods = np.random.randint(skill_period_min, skill_period_max + 1, size=(num_drones,), dtype=np.int64)
            skill_ages = np.zeros((num_drones,), dtype=np.int64)
            pending_update = np.ones((num_drones,), dtype=bool)
            switches = 0
            ep_reward = 0.0
            max_steps = int(float(env.cfg.episode_len_sec) * float(env.cfg.ctrl_freq))

            for t in range(max_steps):
                gobs = build_high_level_obs(info, num_drones, max_sensed, sensing_radius)
                with torch.no_grad():
                    logits_t, _ = high_model.forward(torch.as_tensor(gobs[None, :], dtype=torch.float32, device=device))
                    logits_t = logits_t + skill_mask.view(1, 1, -1)
                    if args.deterministic_high:
                        sampled_np = torch.argmax(logits_t, dim=-1)[0].cpu().numpy().astype(np.int64)
                    else:
                        sampled_np = torch.distributions.Categorical(logits=logits_t).sample()[0].cpu().numpy().astype(np.int64)

                if np.any(pending_update):
                    cbf_now = np.asarray(info["cbf_state"], dtype=np.float32)
                    for i in range(num_drones):
                        if not pending_update[i]:
                            continue
                        candidate = int(sampled_np[i])
                        if not in_skill_initiation_set(cbf_now[i], min_altitude=float(_ckpt_get(saved_args, "skill_init_min_alt", 0.2))):
                            candidate = fallback_skill
                        current_skills[i] = candidate
                        skill_periods[i] = int(np.random.randint(skill_period_min, skill_period_max + 1))
                        skill_ages[i] = 0
                        switches += 1

                desired_now = np.asarray(info["desired_positions"], dtype=np.float32)
                target_now = build_skill_targets(
                    desired_now,
                    current_skills,
                    float(_ckpt_get(saved_args, "delta_x", 0.35)),
                    float(_ckpt_get(saved_args, "delta_y", 0.35)),
                    float(_ckpt_get(saved_args, "delta_z", 0.35)),
                )
                low_obs = build_policy_obs(
                    obs,
                    target_now,
                    use_delta_feature=bool(_ckpt_get(saved_args, "use_delta_feature", True)),
                    normalize_obs=bool(_ckpt_get(saved_args, "normalize_obs", True)),
                    num_drones=num_drones,
                    max_sensed_obstacles=max_sensed,
                    include_neighbor_features=env.cfg.include_neighbor_features,
                    pos_scale=float(_ckpt_get(saved_args, "obs_pos_scale", 5.0)),
                    vel_scale=float(_ckpt_get(saved_args, "obs_vel_scale", 2.0)),
                    clip_val=float(_ckpt_get(saved_args, "obs_clip", 5.0)),
                )
                tau_vec = np.clip(skill_ages / np.maximum(skill_periods - 1, 1), 0.0, 1.0).astype(np.float32)
                cbf_state = np.asarray(info["cbf_state"], dtype=np.float32)
                vel_now = np.asarray(cbf_state[:, 3:6], dtype=np.float32)
                obs_A_now = np.asarray(
                    info.get("obstacle_cbf_A", np.zeros((num_drones, max_sensed, 3), dtype=np.float32)),
                    dtype=np.float32,
                )
                obs_b_now = np.asarray(
                    info.get("obstacle_cbf_b", np.zeros((num_drones, max_sensed), dtype=np.float32)),
                    dtype=np.float32,
                )

                with torch.no_grad():
                    out = low_model.act(
                        obs_local=torch.as_tensor(low_obs[None, ...], dtype=torch.float32, device=device),
                        skill_idx=torch.as_tensor(current_skills[None, ...], dtype=torch.long, device=device),
                        tau=torch.as_tensor(tau_vec[None, :, None], dtype=torch.float32, device=device),
                        cbf_state=torch.as_tensor(cbf_state[None, ...], dtype=torch.float32, device=device),
                        qp_target_pos=torch.as_tensor(target_now[None, ...], dtype=torch.float32, device=device),
                        qp_target_vel=torch.zeros((1, num_drones, 3), dtype=torch.float32, device=device),
                        qp_skill_idx=torch.as_tensor(current_skills[None, ...], dtype=torch.long, device=device),
                        qp_obstacle_A=torch.as_tensor(obs_A_now[None, ...], dtype=torch.float32, device=device),
                        qp_obstacle_b=torch.as_tensor(obs_b_now[None, ...], dtype=torch.float32, device=device),
                        qp_solver=qp_solver,
                        qp_nominal_anchor_weight=u_nom_anchor_w,
                        deterministic=bool(args.deterministic_low),
                    )

                u_safe = out["u_safe"][0].cpu().numpy().astype(np.float32)
                action = env.velocity_to_normalized_action(u_safe)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)

                ref_dir, ref_speed = build_skill_reference_velocity(
                    current_skills,
                    cruise_speed=float(_ckpt_get(saved_args, "skill_cruise_speed", 0.5)),
                    accelerate_speed=float(_ckpt_get(saved_args, "skill_accelerate_speed", 0.2)),
                    decelerate_speed=float(_ckpt_get(saved_args, "skill_decelerate_speed", 0.2)),
                    vel_now_n3=vel_now,
                    speed_cap=env.cfg.max_target_speed_xy,
                )
                cbf_next = np.asarray(info["cbf_state"], dtype=np.float32)
                pos_next = cbf_next[:, 0:3]
                vel_next = cbf_next[:, 3:6]
                desired_next = np.asarray(info["desired_positions"], dtype=np.float32)
                target_next = build_skill_targets(
                    desired_next,
                    current_skills,
                    float(_ckpt_get(saved_args, "delta_x", 0.35)),
                    float(_ckpt_get(saved_args, "delta_y", 0.35)),
                    float(_ckpt_get(saved_args, "delta_z", 0.35)),
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
                    pos_tol=float(_ckpt_get(saved_args, "skill_term_pos_tol", 0.18)),
                    speed_tol=float(_ckpt_get(saved_args, "skill_term_speed_tol", 0.18)),
                    heading_tol=float(_ckpt_get(saved_args, "skill_term_heading_tol", 0.40)),
                    min_duration=int(_ckpt_get(saved_args, "skill_min_duration", 8)),
                    pos_guard_scale=float(_ckpt_get(saved_args, "skill_term_pos_guard_scale", 3.0)),
                )
                pending_update = term_mask.copy()
                skill_ages = age_after

                if (t % max(int(args.log_every), 1)) == 0:
                    mean_x = float(np.mean(pos_next[:, 0]))
                    print(
                        f"ep={ep+1} step={t:04d} mean_x={mean_x:.2f} "
                        f"min_pair={float(info.get('min_pairwise_distance', np.nan)):.3f} "
                        f"min_obs_clear={float(info.get('min_obstacle_clearance', np.nan)):.3f} "
                        f"contacts={int(info.get('obstacle_contact_count', 0))} "
                        f"skill_succ={float(np.mean(succ_mask.astype(np.float32))):.3f} "
                        f"skill_timeout={float(np.mean(tout_mask.astype(np.float32))):.3f}"
                    )

                if args.gui and args.sleep:
                    time.sleep(ctrl_dt)

                if terminated or truncated:
                    break

            print(
                f"[Replay] episode={ep+1}/{args.episodes} total_reward={ep_reward:.3f} "
                f"switches={switches} done={bool(terminated or truncated)}"
            )

        if args.gui and args.keep_gui:
            print("Replay finished. GUI kept alive. Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
