"""Gymnasium CarRacing-v3 adapter for Maxime Burchi's DreamerV3-PyTorch API.

The upstream continuous policy expects symmetric actions in [-1, 1].
We expose a 2-D policy action:
  a[0] = steering in [-1, 1]
  a[1] = signed longitudinal control: positive throttle, negative brake.
It is mapped to Gymnasium's native [steer, gas, brake] action.

Observations are RGB uint8 tensors in CHW layout at the requested image size.
The return contract matches the upstream DMC/Atari adapters:
state, reward, done, is_first, is_last.
"""
from __future__ import annotations

import secrets
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


class AttrDict(dict):
    """Small attribute-access dictionary compatible with upstream wrappers."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def policy_to_carracing_action(action: Any) -> np.ndarray:
    """Map DreamerV3's symmetric 2-D action to CarRacing's native 3-D action."""
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (2,):
        raise ValueError(f"Expected a 2-D policy action, got shape {action.shape}")
    action = np.clip(action, -1.0, 1.0)
    steer = float(action[0])
    longitudinal = float(action[1])
    gas = max(longitudinal, 0.0)
    brake = max(-longitudinal, 0.0)
    return np.asarray([steer, gas, brake], dtype=np.float32)


class CarRacingEnv:
    """Environment class matching the API consumed by upstream DreamerV3."""

    def __init__(
        self,
        task="v3",
        mode="rgb",
        history_frames=1,
        img_size=(64, 64),
        action_repeat=1,
        seed=None,
        render_mode=None,
        max_episode_steps=1000,
    ):
        if task != "v3":
            raise ValueError(f"Only CarRacing-v3 is supported, got task={task!r}")
        if mode != "rgb":
            raise ValueError(f"DreamerV3 CarRacing adapter requires mode='rgb', got {mode!r}")
        if history_frames != 1:
            raise ValueError("This adapter currently supports history_frames=1 only")
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")

        self.img_size = tuple(int(v) for v in img_size)
        self.action_repeat = int(action_repeat)
        # The upstream policy uses symmetric [-1, 1] controls. Two dimensions
        # avoid the neutral-action ambiguity of mapping three symmetric outputs
        # onto CarRacing's [steer, gas, brake] ranges.
        self.num_actions = 2
        self.action_size = (2,)
        self.state_size = (3, self.img_size[0], self.img_size[1])
        self.clip_low = -1.0
        self.clip_high = 1.0
        self.fps = 50
        self.episode_score = 0.0
        self.episode_steps = 0
        self._next_seed = seed

        self.env = gym.make(
            "CarRacing-v3",
            continuous=True,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )

    def _image_tensor(self, observation: np.ndarray) -> torch.Tensor:
        if observation.ndim != 3 or observation.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB observation, got {observation.shape}")
        image = torch.from_numpy(np.asarray(observation).copy()).permute(2, 0, 1).contiguous()
        if tuple(image.shape[-2:]) != self.img_size:
            image = F.interpolate(
                image.unsqueeze(0).to(torch.float32),
                size=self.img_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0).round().clamp_(0, 255).to(torch.uint8)
        return image.contiguous()

    def sample(self) -> torch.Tensor:
        return torch.empty(self.action_size, dtype=torch.float32).uniform_(
            self.clip_low, self.clip_high
        )

    def set_seed(self, seed: int) -> None:
        self._next_seed = int(seed)

    def random_seed(self) -> None:
        self._next_seed = secrets.randbelow(2**31 - 1)

    def reset(self) -> AttrDict:
        if self._next_seed is None:
            observation, _ = self.env.reset()
        else:
            observation, _ = self.env.reset(seed=self._next_seed)
            self._next_seed = None

        self.episode_score = 0.0
        self.episode_steps = 0
        return AttrDict(
            state=self._image_tensor(observation),
            reward=torch.tensor(0.0, dtype=torch.float32),
            done=torch.tensor(False, dtype=torch.float32),
            is_first=torch.tensor(True, dtype=torch.float32),
            is_last=torch.tensor(False, dtype=torch.float32),
        )

    def step(self, action: Any) -> AttrDict:
        native_action = policy_to_carracing_action(action)
        total_reward = 0.0
        terminated = False
        truncated = False
        observation = None

        for _ in range(self.action_repeat):
            observation, reward, terminated, truncated, _info = self.env.step(native_action)
            total_reward += float(reward)
            self.episode_steps += 1
            if terminated or truncated:
                break

        self.episode_score += total_reward
        is_last = bool(terminated or truncated)
        return AttrDict(
            state=self._image_tensor(observation),
            reward=torch.tensor(total_reward, dtype=torch.float32),
            # True termination stops value bootstrapping; time-limit truncation
            # ends the episode but keeps the bootstrap target valid.
            done=torch.tensor(bool(terminated), dtype=torch.float32),
            is_first=torch.tensor(False, dtype=torch.float32),
            is_last=torch.tensor(is_last, dtype=torch.float32),
        )

    def get_episode_state(self) -> dict:
        return {
            "episode_steps": int(self.episode_steps),
            "episode_reward_total": float(self.episode_score),
        }

    def close(self) -> None:
        self.env.close()
