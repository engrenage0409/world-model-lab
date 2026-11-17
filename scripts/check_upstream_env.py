"""Check pinned V1/V2/V3 source trees and smoke-test Gymnasium CarRacing.

Does not import old upstream dependencies, train a model, or modify checkouts.
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "baselines" / "sources.json"


def check_sources():
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))["sources"]
    success = True
    for entry in sources:
        directory = ROOT / "external" / entry["folder"]
        if not directory.is_dir():
            print(f"[MISSING] {entry['version']}: {directory}")
            success = False
            continue
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False)
        sha = result.stdout.strip()
        matched = result.returncode == 0 and sha == entry["commit"]
        print(f"[{'OK' if matched else 'MISMATCH'}] {entry['version']}: "
              f"{sha[:12] or result.stderr.strip()}")
        success = success and matched
    return success


def smoke_carracing(steps):
    import gymnasium as gym
    import numpy as np
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    try:
        obs, info = env.reset(seed=123)
        print(f"[ENV] obs={obs.shape}, dtype={obs.dtype}; "
              f"action={env.action_space.shape}, bounds="
              f"{env.action_space.low}..{env.action_space.high}")
        if obs.shape != (96, 96, 3) or obs.dtype != np.uint8:
            raise RuntimeError("Unexpected CarRacing RGB observation schema")
        if env.action_space.shape != (3,):
            raise RuntimeError("Expected continuous [steer, gas, brake]")
        total = 0.0
        for step in range(steps):
            action = np.array([0.0, 0.35, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            if terminated or truncated:
                print(f"[ENV] episode ended at step {step + 1}")
                break
        print(f"[OK] CarRacing smoke test; transitions={step + 1}, "
              f"return={total:.3f}; terminated={terminated}, truncated={truncated}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--skip-env", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    pinned = check_sources()
    if not args.skip_env:
        smoke_carracing(args.steps)
    if not pinned:
        print("[WARN] One or more upstream source trees are missing or mismatched.")
    print("This checks checkouts and Gymnasium only; it is NOT an upstream Dreamer training test.")


if __name__ == "__main__":
    main()
