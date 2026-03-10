import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drone_env import (
    DifferentiableCBFQPConfig,
    FormationAviaryEnv,
    FormationEnvConfig,
    RLCBFQPWrapper,
    build_solver_from_velocity_bounds,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-drones", type=int, default=3)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    base_env = FormationAviaryEnv(FormationEnvConfig(num_drones=args.num_drones, gui=args.gui))
    solver = build_solver_from_velocity_bounds(
        num_drones=args.num_drones,
        max_speed_xy=base_env.cfg.max_target_speed_xy,
        max_speed_z=base_env.cfg.max_target_speed_z,
        config=DifferentiableCBFQPConfig(safe_distance=base_env.cfg.safe_distance),
        device=args.device,
    )
    env = RLCBFQPWrapper(base_env, qp_solver=solver)

    obs, info = env.reset(seed=42)
    print("obs_shape=", obs.shape)
    print("cbf_state_shape=", info["cbf_state"].shape)

    total_reward = 0.0
    for k in range(args.steps):
        # desired velocity command in m/s
        action = 0.7 * np.random.randn(args.num_drones, 3).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if k % 25 == 0:
            qpi = info.get("qp_info", {})
            print(
                f"step={k:04d} reward={reward:.3f} track={info['mean_track_error']:.3f} "
                f"min_d={info['min_pairwise_distance']:.3f} qp={qpi.get('status', 'none')} "
                f"slack={qpi.get('slack_l2', 0.0):.4f}"
            )

        if terminated or truncated:
            print(f"episode_end terminated={terminated} truncated={truncated} step={k}")
            break

        if args.gui:
            time.sleep(1.0 / env.unwrapped.cfg.ctrl_freq)

    print("total_reward=", round(total_reward, 3))
    env.close()


if __name__ == "__main__":
    main()
