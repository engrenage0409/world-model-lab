import pytest
import torch

from world_model.vision_losses import image_objective, edge_l1_loss, ssim_loss
from world_model.rssm import RSSMWorldModel
from world_model.categorical_rssm import CategoricalRSSM


def test_objectives_identical_frames_and_gradient():
    torch.manual_seed(3)
    target = torch.rand(2, 3, 16, 16)
    same = image_objective(target, target, mode="l1_ssim_edge")
    assert same.item() < 1e-6
    assert edge_l1_loss(target, target).item() == 0
    assert ssim_loss(target, target).item() < 1e-6
    estimate = torch.rand(2, 3, 16, 16, requires_grad=True)
    objective = image_objective(estimate, target, "l1_ssim_edge")
    assert objective.isfinite() and objective.item() > 0
    objective.backward()
    assert estimate.grad is not None
    assert torch.isfinite(estimate.grad).all()


@pytest.mark.parametrize("model", [
    RSSMWorldModel(hidden=32, stoch=8, embed=32),
    CategoricalRSSM(hidden=32, groups=4, classes=4, embed=32)
])
def test_models_support_edge_aware_loss(model):
    torch.manual_seed(5)
    observations = torch.rand(1, 3, 3, 96, 96)
    actions = torch.zeros(1, 2, 3)
    rewards = torch.zeros(1, 2)
    dones = torch.zeros(1, 2)
    metrics = model.loss(
        observations, actions, rewards, dones,
        recon_mode="l1_ssim_edge")
    assert torch.isfinite(metrics["loss"])
    assert torch.isfinite(metrics["recon_mse"])
    assert torch.isfinite(metrics["recon_objective"])
    metrics["loss"].backward()
    assert model.decoder.net[0].weight.grad is not None


def test_invalid_objective():
    a = torch.zeros(1, 3, 4, 4)
    with pytest.raises(ValueError):
        image_objective(a, a, mode="unknown")
    with pytest.raises(ValueError):
        image_objective(a, a[:, :, :3], mode="mse")
