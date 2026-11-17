import torch

from world_model.categorical_rssm import CategoricalRSSM
from world_model.checkpoints import model_from_checkpoint


def test_categorical_loss_and_gradients():
    torch.manual_seed(7)
    model = CategoricalRSSM(hidden=32, groups=4, classes=4, embed=32)
    observations = torch.rand(1, 3, 3, 96, 96)
    actions = torch.rand(1, 2, 3)
    rewards = torch.zeros(1, 2)
    dones = torch.zeros(1, 2)
    output = model.loss(observations, actions, rewards, dones, free_nats=0.0)
    assert set(("loss", "dyn_kl", "rep_kl", "recon_mse")).issubset(output)
    assert all(torch.isfinite(value) for value in output.values())
    output["loss"].backward()
    assert model.gru.weight_ih.grad is not None
    assert model.prior_net[-1].weight.grad is not None
    assert model.posterior_net[-1].weight.grad is not None
    assert model.decoder.net[0].weight.grad is not None


def test_categorical_prior_shape_and_onehot():
    model = CategoricalRSSM(hidden=32, groups=4, classes=4, embed=32)
    start = torch.rand(1, 3, 96, 96)
    actions = torch.zeros(1, 2, 3)
    state = model.observe_first(start)
    next_state, distribution = model.prior_step(state, actions[:, 0], sample=False)
    assert next_state.z.shape == (1, 4, 4)
    assert torch.allclose(next_state.z.sum(-1), torch.ones(1, 4))
    assert torch.allclose(distribution.probs.sum(-1), torch.ones(1, 4))
    predicted, rewards = model.predict_open_loop(start, actions)
    assert predicted.shape == (1, 2, 3, 96, 96)
    assert rewards.shape == (1, 2)


def test_checkpoint_loads_both_variants():
    from world_model.rssm import RSSMWorldModel
    for kind, config, cls in [
        ("gaussian", dict(hidden=32, stoch=8, embed=32), RSSMWorldModel),
        ("categorical", dict(hidden=32, groups=4, classes=4, embed=32), CategoricalRSSM),
    ]:
        original = cls(**config)
        ckpt = {"model": original.state_dict(), "config": config}
        if kind == "categorical":
            ckpt["model_type"] = kind
        restored = model_from_checkpoint(ckpt)
        assert isinstance(restored, cls)
