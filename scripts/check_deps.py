import importlib
import sys

PKGS = [
    ("numpy", "numpy"),
    ("gymnasium", "gymnasium"),
    ("pybullet", "pybullet"),
    ("gym_pybullet_drones", "gym-pybullet-drones"),
    ("transforms3d", "transforms3d"),
    ("PIL", "pillow"),
    ("pkg_resources", "setuptools<81"),
]

failed = []
for module_name, install_name in PKGS:
    try:
        importlib.import_module(module_name)
        print(f"[OK] {module_name} (pip/conda: {install_name})")
    except Exception as e:
        print(f"[MISSING] {module_name} (pip/conda: {install_name}): {e}")
        failed.append(module_name)

print("python=", sys.executable)
if failed:
    raise SystemExit(1)
