import torch

from world_model.v3_rssm import V3RSSM, symlog, symexp, twohot
from world_model.checkpoints import model_from_checkpoint


def test_symlog_inverse_and_twohot():
    values = torch.tensor([-10.0, -1.0, 0.0, 2.0, 10.0])
    assert torch.allclose(symexp(symlog(values)), values, atol=1e-5)
    centers = torch.linspace(-3, 3, 7)
    distribution = twohot(torch.tensor([-9., -2.5, 0., 1.5, 9.]), centers)
    assert torch.allclose(distribution.sum(-1), torch.ones(5))
    assert torch.all(distribution >= 0)


def test_v3_model_shapes_gradients_checkpoint():
    torch.manual_seed(1)
    config = dict(hidden=32, groups=4, classes=4, embed=32, reward_bins=21)
    model = V3RSSM(**config)
    obs = torch.rand(1, 3, 3, 96, 96)
    actions = torch.rand(1, 2, 3)
    rewards = torch.tensor([[1.0, -0.1]])
    dones = torch.zeros(1, 2)
    result = model.loss(obs, actions, rewards, dones)
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["reward_twohot_ce"])
    result["loss"].backward()
    assert model.reward_head[-1].weight.grad is not None
    assert model.encoder.net[0].weight.grad is not None
    future, pred_reward = model.predict_open_loop(obs[:, 0], actions)
    assert future.shape == (1, 2, 3, 96, 96)
    assert pred_reward.shape == (1, 2)
    ckpt = {"model_type": "v3_experimental", "config": config,
            "model": model.state_dict()}
    restored = model_from_checkpoint(ckpt)
    assert isinstance(restored, V3RSSM)
