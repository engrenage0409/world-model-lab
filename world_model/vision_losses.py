"""Differentiable image objectives for CarRacing reconstruction ablations.

All images are float RGB [B,3,H,W] normalized to [0,1].
SSIM uses a local 3x3 average window, so this is an experimental lightweight
SSIM surrogate rather than a standard Gaussian-window benchmark metric.
"""
import torch
import torch.nn.functional as F


def ssim_loss(pred, target):
    """Mean (1 - local SSIM) / 2, in [0,1] up to numerical precision."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    var_x = (F.avg_pool2d(pred.square(), 3, 1, 1) - mu_x.square()).clamp_min(0)
    var_y = (F.avg_pool2d(target.square(), 3, 1, 1) - mu_y.square()).clamp_min(0)
    cov = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    similarity = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2))
    return ((1 - similarity.clamp(-1, 1)) / 2).mean()


def edge_l1_loss(pred, target):
    """Compare image gradients along X and Y, including road boundaries."""
    pred_dx, target_dx = pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1]
    pred_dy, target_dy = pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    return (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)) / 2


def image_objective(pred, target, mode="mse", ssim_weight=0.2, edge_weight=0.1):
    """Return a differentiable reconstruction objective, not an evaluation metric."""
    if pred.shape != target.shape:
        raise ValueError("Reconstruction and target shapes differ")
    if mode == "mse":
        return F.mse_loss(pred, target)
    if mode == "l1_ssim_edge":
        if ssim_weight < 0 or edge_weight < 0:
            raise ValueError("Loss weights must be nonnegative")
        return (F.l1_loss(pred, target)
                + ssim_weight * ssim_loss(pred, target)
                + edge_weight * edge_l1_loss(pred, target))
    raise ValueError(f"Unsupported reconstruction mode: {mode}")
