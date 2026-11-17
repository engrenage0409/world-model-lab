"""Small, reproducible DreamerV3-CarRacing config using the upstream model unchanged.

Run from external/dreamer_v3 with the project root on PYTHONPATH.
"""
import json
import os
from pathlib import Path

import torch
import nnet
from nnet.envs import dm_control

from integrations.dreamerv3_carracing.env import CarRacingEnv

# Register our adapter in the upstream environment registry. The upstream
# DreamerV3 model and its World Model / Actor / Critic code remain unchanged.
dm_control.dm_control_dict["CarRacing"] = CarRacingEnv

env_name = os.environ.get("env_name", "dmc-CarRacing-v3")
if env_name != "dmc-CarRacing-v3":
    raise ValueError(f"Expected env_name=dmc-CarRacing-v3, got {env_name!r}")

default_override = {
    "env_params": {
        "task": "v3",
        "mode": "rgb",
        "history_frames": 1,
        "img_size": [64, 64],
        "action_repeat": 1,
    },
    "time_limit": 1000,
    "batch_size": 2,
    "L": 8,
    "H": 5,
    "num_envs": 1,
    "epochs": 1,
    "epoch_length": 1,
    "env_step_period": 8,
    "pre_fill_steps": 16,
    "buffer_capacity": 5000,
    "eval_epidodes": 0,
    "parallel_envs": False,
    "thread_env_step": False,
    # XS-sized network dimensions for the first end-to-end smoke run.
    "model_deter_size": 256,
    "dim_cnn": 24,
    "repr_hidden_size": 256,
    "model_hidden_size": 256,
    "action_hidden_size": 256,
    "value_hidden_size": 256,
    "reward_hidden_size": 256,
    "discount_hidden_size": 256,
    "repr_layers": 1,
    "action_layers": 1,
    "value_layers": 1,
    "reward_layers": 1,
    "discount_layers": 1,
}

override_config = os.environ.get("override_config")
if override_config:
    override_config = json.loads(override_config)
else:
    override_config = default_override

model = nnet.models.rl.dreamer.DreamerV3(
    env_name=env_name,
    override_config=override_config,
)
if os.environ.get("DREAMER_PRECISION", "float32").lower() == "float32":
    model.config.precision = torch.float32
model.compile()

precision = model.config.precision
grad_init_scale = model.config.grad_init_scale
epochs = model.config.epochs
epoch_length = model.config.epoch_length
save_trajectories = True

project_root = Path(__file__).resolve().parents[2]
run_name = os.environ.get("run_name", "carracing-smoke")
callback_path = str(project_root / "results" / "dreamerv3_carracing" / run_name)

training_dataset = nnet.datasets.replay_buffers.DreamerV3ReplayBuffer(
    batch_size=model.config.batch_size,
    root=callback_path,
    buffer_capacity=model.config.buffer_capacity,
    epoch_length=epoch_length,
    sample_length=model.config.L,
    collate_fn=model.config.collate_fn,
    save_trajectories=save_trajectories,
)
model.set_replay_buffer(training_dataset)

evaluation_dataset = nnet.datasets.VoidDataset(
    num_steps=model.config.eval_epidodes
)
