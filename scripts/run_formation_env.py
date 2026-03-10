import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drone_env import FormationAviaryEnv, FormationEnvConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-drones", type=int, default=3)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    env = FormationAviaryEnv(FormationEnvConfig(num_drones=args.num_drones, gui=args.gui))
    obs, info = env.reset(seed=42)
    print("obs_shape=", obs.shape)
    print("init_cbf_margin=", info["cbf_margin"])

    total_reward = 0.0
    for k in range(args.steps):
        action = 0.2 * np.random.randn(args.num_drones, 3).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if k % 50 == 0:
            print(
                f"step={k:04d} reward={reward:.3f} "
                f"track={info['mean_track_error']:.3f} "
                f"min_d={info['min_pairwise_distance']:.3f} "
                f"cbf_margin={info['cbf_margin']:.3f}"
            )

        if terminated or truncated:
            print(f"episode_end terminated={terminated} truncated={truncated} step={k}")
            break

        if args.gui:
            time.sleep(1.0 / env.cfg.ctrl_freq)

    print("total_reward=", round(total_reward, 3))
    env.close()


if __name__ == "__main__":
    main()
