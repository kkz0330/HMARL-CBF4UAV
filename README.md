# UAV Safe Formation (Step-1 Simulator)

This project sets up the low-level simulation layer for UAV safe formation using `gym-pybullet-drones`.

## Project Structure

- `drone_env/formation_env.py`: gymnasium environment wrapper (`FormationAviaryEnv`)
- `scripts/check_deps.py`: dependency checker
- `scripts/run_formation_env.py`: rollout smoke test
- `requirements.txt`: pinned runtime dependencies

## Recommended Environment

- Python: `3.10` (tested with `D:\anaconda\envs\drones310`)
- OS: Windows

## Install (Validated on this machine)

```powershell
# 1) install pybullet from conda-forge (avoids local C++ build)
D:\anaconda\Scripts\conda.exe install -n drones310 -c conda-forge -y pybullet

# 2) install Python dependencies
& "D:\anaconda\envs\drones310\python.exe" -m pip install -r .\requirements.txt
```

## Verify and Run

```powershell
cd /d "D:\uav_safe_formation"
& "D:\anaconda\envs\drones310\python.exe" .\scripts\check_deps.py
& "D:\anaconda\envs\drones310\python.exe" .\scripts\run_formation_env.py --num-drones 3 --steps 200
```

## RL + CBF-QP Wrapper

`drone_env/rl_cbf_wrapper.py` provides `RLCBFQPWrapper`:

- PPO-facing action: `(N, 3)` desired velocity in m/s
- Optional safety filter hook: `qp_solver(cbf_state, v_des, last_info) -> (v_safe, qp_info)`
- Unified `info` keys: `v_des`, `v_safe`, `normalized_action`, `qp_info`, `cbf_state`

`drone_env/differentiable_cbf_qp.py` provides a differentiable multi-UAV CBF-QP solver
that can be injected directly into `RLCBFQPWrapper`.

Quick demo:

```powershell
& "D:\anaconda\envs\drones310\python.exe" .\scripts\run_with_diff_cbf_qp.py --num-drones 3 --steps 100
```

## Notes

- `gym-pybullet-drones` currently imports `pkg_resources`, so `setuptools<81` is pinned.
- If you hit Windows native crash `0xc06d007f` around `numpy.linalg.inv`, reinstall NumPy wheel in the target env:

```powershell
& "D:\anaconda\envs\drones310\python.exe" -m pip install --force-reinstall --no-cache-dir numpy==2.2.6
```

- The current env wrapper exposes safety-aligned info fields (e.g., `min_pairwise_distance`, `cbf_margin`) for your next CBF/QP integration step.
