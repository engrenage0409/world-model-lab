"""Collect diverse CarRacing-v3 episodes without overwriting existing data.

Policies:
 random: independent per-step controls (legacy style)
 smooth: temporally-correlated steering, throttle and occasional braking
 mixed: reproducibly alternate the two per episode
This is exploration data, NOT a demonstrated expert or a trained driver.
"""
import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np


def new_action(rng, mode, previous=None, step=0):
    if mode == "random":
        return np.array([rng.uniform(-0.7, 0.7),
                         rng.uniform(0.1, 0.8), 0.0], dtype=np.float32)
    if mode != "smooth":
        raise ValueError(f"Unsupported policy: {mode}")
    if previous is None:
        previous = np.array([0.0, 0.55, 0.0], dtype=np.float32)
    # Smooth actions produce sustained turns rather than random jitter.
    steer = np.clip(0.94 * previous[0] + rng.normal(0, 0.085), -0.9, 0.9)
    gas = np.clip(0.95 * previous[1] + 0.05 * rng.uniform(0.3, 0.85), 0.1, 0.9)
    # Short sparse braking events. May not keep a vehicle on the road.
    brake = rng.uniform(0.1, 0.45) if rng.random() < 0.015 else max(0.0, previous[2] * 0.5)
    if brake > 0.05:
        gas = min(gas, 0.15)
    return np.array([steer, gas, brake], dtype=np.float32)


def collect(args):
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("Episodes and max-steps must be positive")
    if args.start_index < 0:
        raise ValueError("start-index must be nonnegative")

    # Preflight: avoid half a run before discovering an existing file.
    planned = [output / f"episode_{idx:04d}.npz"
               for idx in range(args.start_index, args.start_index + args.episodes)]
    existing = [path for path in planned if path.exists()]
    if existing:
        raise FileExistsError(
            f"{len(existing)} episode files already exist (first: {existing[0]}). "
            "Choose a new --start-index; no files were changed.")

    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    try:
        append = manifest_path.exists()
        with manifest_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "file", "seed", "policy", "steps", "return",
                "steer_mean", "steer_std", "gas_mean", "brake_fraction",
                "terminated", "truncated"])
            if not append:
                writer.writeheader()
            for index in range(args.start_index, args.start_index + args.episodes):
                seed = args.seed + index
                rng = np.random.default_rng(seed)
                mode = args.policy
                if mode == "mixed":
                    mode = "random" if index % 2 == 0 else "smooth"
                obs, _ = env.reset(seed=seed)
                observations = [obs]
                actions, rewards, dones = [], [], []
                previous = None
                terminated, truncated = False, False
                for step in range(args.max_steps):
                    action = new_action(rng, mode, previous, step)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
                    actions.append(action)
                    rewards.append(float(reward))
                    dones.append(done)
                    observations.append(next_obs)
                    previous = action
                    obs = next_obs
                    if done:
                        break
                arr = np.asarray(actions, dtype=np.float32)
                path = output / f"episode_{index:04d}.npz"
                np.savez_compressed(
                    path, observations=np.asarray(observations, dtype=np.uint8),
                    actions=arr, rewards=np.asarray(rewards, dtype=np.float32),
                    dones=np.asarray(dones, dtype=bool))
                row = {
                    "file": path.name, "seed": seed, "policy": mode,
                    "steps": len(actions), "return": round(sum(rewards), 4),
                    "steer_mean": round(float(arr[:, 0].mean()), 4),
                    "steer_std": round(float(arr[:, 0].std()), 4),
                    "gas_mean": round(float(arr[:, 1].mean()), 4),
                    "brake_fraction": round(float((arr[:, 2] > 0.01).mean()), 4),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                }
                writer.writerow(row)
                handle.flush()
                print(f"{path}: policy={mode} steps={len(actions)} return={row['return']}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--policy", choices=["random", "smooth", "mixed"], default="mixed")
    parser.add_argument("--out", default="data/episodes_v2")
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
