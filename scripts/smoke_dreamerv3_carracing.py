"""Construct the pinned upstream DreamerV3 model and smoke-test its env wrapper."""
import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "dreamer_v3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if not UPSTREAM.is_dir():
        raise FileNotFoundError(f"Missing upstream checkout: {UPSTREAM}")

    # Upstream nnet package resolves from its pinned checkout; integration
    # modules resolve from the project root.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(UPSTREAM))
    os.environ["env_name"] = "dmc-CarRacing-v3"
    os.environ["run_name"] = "adapter-smoke"
    os.environ["DREAMER_PRECISION"] = "float32"
    os.environ.pop("override_config", None)

    config = importlib.import_module("integrations.dreamerv3_carracing.config")
    model = config.model
    print("Upstream model:", model.__class__.__module__, model.__class__.__name__)
    print("Policy action dimension:", model.env.num_actions)
    print("Observation batch shape:", tuple(model.env.reset().state.shape))

    try:
        for index in range(args.steps):
            action = model.env.sample()
            obs = model.env.step(action)
            print(
                f"step={index + 1} state={tuple(obs.state.shape)} "
                f"reward={obs.reward.tolist()} done={obs.done.tolist()} "
                f"is_last={obs.is_last.tolist()} error={obs.error.tolist()}"
            )
            if bool(obs.is_last[0]):
                model.env.reset(env_i=0)
    finally:
        for wrapped in model.env.envs:
            current = wrapped
            while hasattr(current, "env"):
                current = current.env
            if hasattr(current, "close"):
                current.close()


if __name__ == "__main__":
    main()
