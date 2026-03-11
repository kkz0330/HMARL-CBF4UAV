import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drone_env import LocalObstacleEnvConfig, LocalObstacleFormationEnv


def main() -> None:
    # Single-drone sanity check in obstacle-free scene.
    env = LocalObstacleFormationEnv(
        LocalObstacleEnvConfig(
            num_drones=1,
            scenario="none",
            gui=False,
            episode_len_sec=20.0,
        )
    )

    target = np.array([1.5, 0.0, 1.0], dtype=np.float32)

    try:
        obs, info = env.reset(seed=42)
        print("target:", target.tolist())
        print("init_pos:", info["cbf_state"][0, 0:3].tolist())

        for step in range(500):
            current_pos = np.asarray(info["cbf_state"][0, 0:3], dtype=np.float32)
            delta_p = target - current_pos
            dist = float(np.linalg.norm(delta_p))

            # "Perfect" normalized action direction in [-1, 1].
            direction = delta_p / (dist + 1e-6)
            perfect_action = np.clip(direction * 0.5, -1.0, 1.0).astype(np.float32)
            action_input = perfect_action.reshape(1, 3)

            obs, reward, terminated, truncated, info = env.step(action_input)

            print(
                f"step={step:03d} action={action_input[0]} pos={current_pos} "
                f"dist={dist:.3f} reward={reward:.4f}"
            )

            if dist < 0.10:
                print("Reached target neighborhood (<0.10 m).")
                break
            if terminated or truncated:
                print(f"Episode ended terminated={terminated} truncated={truncated} at step={step}.")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()

