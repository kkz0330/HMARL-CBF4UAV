import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drone_env import LocalObstacleEnvConfig, LocalObstacleFormationEnv
from drone_env import CBFQPSafetyFilter, CBFQPSafetyFilterConfig, RLCBFQPWrapper


def build_line_offsets(num_drones: int, spacing: float) -> np.ndarray:
    center = 0.5 * (num_drones - 1)
    offsets = np.zeros((num_drones, 3), dtype=np.float32)
    for i in range(num_drones):
        offsets[i, 1] = (i - center) * spacing
    return offsets


def scripted_action(
    env: LocalObstacleFormationEnv,
    obs: np.ndarray,
    step_idx: int,
    total_steps: int,
    target_x: float = 5.0,
) -> np.ndarray:
    n = env.cfg.num_drones
    pos = obs[:, 0:3]
    vel = obs[:, 3:6]
    dt = 1.0 / float(env.cfg.ctrl_freq)
    t = step_idx * dt

    # Virtual leader moves along +x; this creates a deterministic obstacle-crossing trajectory.
    leader_x = min(target_x, 0.95 * t)
    leader = np.array([leader_x, 0.0, env.cfg.desired_height], dtype=np.float32)
    targets = leader[None, :] + build_line_offsets(n, env.cfg.formation_spacing)

    # Bridge crossing maneuver: compress lateral span and use vertical staggering.
    # Default bridge gap is narrow, so flat line formation cannot pass.
    if 1.1 <= leader_x <= 2.5:
        targets[:, 1] = np.linspace(-0.12, 0.12, n, dtype=np.float32)
        z_offsets = np.where((np.arange(n) % 2) == 0, 0.22, -0.10).astype(np.float32)
        targets[:, 2] = env.cfg.desired_height + z_offsets

    # Tree crossing maneuver: split into two side groups in y.
    # This avoids the trunk/canopy region around x ~= 3.4, y ~= 0.
    if 2.6 <= leader_x <= 4.3:
        side_shift = 0.85
        for i in range(n):
            sign = -1.0 if i < (n // 2) else 1.0
            targets[i, 1] += sign * side_shift

    # Reform after passing the obstacle.
    if leader_x > 4.3:
        pass

    # Simple velocity tracking in Cartesian space -> desired velocity (m/s).
    kp = np.array([1.2, 1.4, 1.8], dtype=np.float32)
    kd = np.array([0.15, 0.20, 0.25], dtype=np.float32)
    v_cmd = kp[None, :] * (targets - pos) - kd[None, :] * vel

    # Add forward cruise if there is still a long way to go.
    remaining = target_x - np.mean(pos[:, 0])
    if remaining > 0.5:
        v_cmd[:, 0] += 0.55

    # Slow down at the tail of the rollout.
    if step_idx > int(0.85 * total_steps):
        v_cmd *= 0.6

    v_xy_max = float(env.cfg.max_target_speed_xy)
    v_z_max = float(env.cfg.max_target_speed_z)
    v_cmd[:, 0:2] = np.clip(v_cmd[:, 0:2], -v_xy_max, v_xy_max)
    v_cmd[:, 2] = np.clip(v_cmd[:, 2], -v_z_max, v_z_max)
    return v_cmd.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-drones", type=int, default=4)
    parser.add_argument("--scenario", type=str, default="bridge_tree", choices=["none", "bridge", "tree", "bridge_tree"])
    parser.add_argument("--sensing-radius", type=float, default=3.0)
    parser.add_argument("--max-sensed-obstacles", type=int, default=3)
    parser.add_argument("--bridge-offset-y", type=float, default=0.40)
    parser.add_argument("--policy", type=str, default="scripted", choices=["scripted", "random"])
    parser.add_argument("--use-cbf-qp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cbf-alpha", type=float, default=4.0)
    parser.add_argument("--cbf-slack-weight", type=float, default=120.0)
    parser.add_argument("--enable-clf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clf-rate", type=float, default=1.0)
    parser.add_argument("--clf-deadzone", type=float, default=0.05)
    parser.add_argument("--clf-slack-weight", type=float, default=20.0)
    parser.add_argument("--episode-len-sec", type=float, default=20.0)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--keep-gui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--manual-replay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    cfg = LocalObstacleEnvConfig(
        num_drones=args.num_drones,
        gui=args.gui,
        scenario=args.scenario,
        sensing_radius=args.sensing_radius,
        max_sensed_obstacles=args.max_sensed_obstacles,
        bridge_pillar_offset_y=args.bridge_offset_y,
        episode_len_sec=args.episode_len_sec,
    )
    base_env = LocalObstacleFormationEnv(cfg)
    ctrl_freq = float(base_env.cfg.ctrl_freq)

    if args.use_cbf_qp:
        vel_low = np.array(
            [-cfg.max_target_speed_xy, -cfg.max_target_speed_xy, -cfg.max_target_speed_z],
            dtype=np.float32,
        )
        vel_high = np.array(
            [cfg.max_target_speed_xy, cfg.max_target_speed_xy, cfg.max_target_speed_z],
            dtype=np.float32,
        )
        qp_solver = CBFQPSafetyFilter(
            num_drones=cfg.num_drones,
            vel_low=vel_low,
            vel_high=vel_high,
            config=CBFQPSafetyFilterConfig(
                alpha=args.cbf_alpha,
                safe_distance=cfg.safe_distance,
                slack_weight=args.cbf_slack_weight,
                enforce_obstacle_constraints=True,
                enable_clf=args.enable_clf,
                clf_rate=args.clf_rate,
                clf_deadzone=args.clf_deadzone,
                clf_slack_weight=args.clf_slack_weight,
            ),
        )
        env = RLCBFQPWrapper(base_env, qp_solver=qp_solver)
    else:
        env = base_env

    try:
        for ep in range(args.episodes):
            if ep > 0 and args.manual_replay:
                cmd = input("Press Enter to replay next episode (or q to quit): ").strip().lower()
                if cmd in {"q", "quit", "exit"}:
                    break

            obs, info = env.reset(seed=42 + ep)
            print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
            print("obs_shape=", obs.shape)
            print("obstacles_total=", info["obstacles_total"])
            print("bridge_inner_gap=", round(2.0 * (cfg.bridge_pillar_offset_y - cfg.bridge_pillar_half_y), 3))
            print("sensed_counts=", info["sensed_obstacle_count"].tolist())
            print("A_obs_shape=", info["obstacle_cbf_A"].shape, "b_obs_shape=", info["obstacle_cbf_b"].shape)

            total_reward = 0.0
            success_cross_tree = False
            min_obs_clearance = float("inf")
            had_obstacle_contact = False

            for k in range(args.steps):
                if args.policy == "random":
                    v_des = 0.35 * np.random.randn(args.num_drones, 3).astype(np.float32)
                else:
                    v_des = scripted_action(base_env, obs, k, args.steps)

                if args.use_cbf_qp:
                    action = v_des
                else:
                    # Base env expects normalized action in [-1,1].
                    action = np.zeros_like(v_des, dtype=np.float32)
                    action[:, 0:2] = v_des[:, 0:2] / float(cfg.max_target_speed_xy)
                    action[:, 2] = v_des[:, 2] / float(cfg.max_target_speed_z)
                    action = np.clip(action, -1.0, 1.0)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                min_obs_clearance = min(min_obs_clearance, float(info["min_obstacle_clearance"]))
                had_obstacle_contact = had_obstacle_contact or (int(info.get("obstacle_contact_count", 0)) > 0)

                mean_x = float(np.mean(obs[:, 0]))
                if mean_x >= 4.4:
                    success_cross_tree = True

                if k % 50 == 0:
                    qp_info = info.get("qp_info", {})
                    v_dev = 0.0
                    if info.get("v_des") is not None and info.get("v_safe") is not None:
                        v_des_i = np.asarray(info["v_des"], dtype=np.float32)
                        v_safe_i = np.asarray(info["v_safe"], dtype=np.float32)
                        v_dev = float(np.linalg.norm(v_safe_i - v_des_i))
                    print(
                        f"step={k:04d} reward={reward:.3f} track={info['mean_track_error']:.3f} mean_x={mean_x:.3f} "
                        f"min_pair_d={info['min_pairwise_distance']:.3f} min_obs_clear={info['min_obstacle_clearance']:.3f} "
                        f"contacts={info.get('obstacle_contact_count', 0)} sensed={info['sensed_obstacle_count'].tolist()} "
                        f"qp={qp_info.get('status', 'off')} "
                        f"slack={qp_info.get('slack_l2', 0.0):.4f} v_dev={v_dev:.4f}"
                    )

                if terminated or truncated:
                    print(f"episode_end terminated={terminated} truncated={truncated} step={k}")
                    break

                if args.gui:
                    time.sleep(1.0 / ctrl_freq)

            print("total_reward=", round(total_reward, 3))
            print("cross_tree_success=", success_cross_tree)
            print("rollout_min_obstacle_clearance=", round(min_obs_clearance, 4))
            print("had_obstacle_contact=", had_obstacle_contact)

        if args.gui and args.keep_gui:
            print("Rollout finished. GUI kept alive. Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
