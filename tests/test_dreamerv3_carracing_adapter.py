import numpy as np
import pytest
import torch

from integrations.dreamerv3_carracing.env import (
    CarRacingEnv,
    policy_to_carracing_action,
)


def test_policy_action_mapping():
    np.testing.assert_allclose(
        policy_to_carracing_action(torch.tensor([0.25, 0.8])),
        np.array([0.25, 0.8, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        policy_to_carracing_action(np.array([-0.4, -0.6])),
        np.array([-0.4, 0.0, 0.6], dtype=np.float32),
    )


def test_policy_action_clips_and_checks_shape():
    np.testing.assert_allclose(
        policy_to_carracing_action([2.0, -2.0]),
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )
    with pytest.raises(ValueError):
        policy_to_carracing_action([0.1, 0.2, 0.3])


def test_gymnasium_adapter_contract():
    env = CarRacingEnv(task="v3", mode="rgb", history_frames=1,
                       img_size=(64, 64), action_repeat=1, seed=123)
    try:
        first = env.reset()
        assert first.state.shape == (3, 64, 64)
        assert first.state.dtype == torch.uint8
        assert first.reward.item() == 0.0
        assert first.is_first.item() == 1.0
        assert first.is_last.item() == 0.0

        next_obs = env.step(torch.tensor([0.0, 0.5]))
        assert next_obs.state.shape == (3, 64, 64)
        assert next_obs.state.dtype == torch.uint8
        assert torch.isfinite(next_obs.reward)
        assert next_obs.is_first.item() == 0.0
    finally:
        env.close()
