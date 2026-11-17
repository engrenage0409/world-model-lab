"""Experimental V3-style RSSM world model.

Adds symlog/two-hot reward prediction to the discrete RSSM. This remains a
partial DreamerV3 implementation: actor/critic, value twohot, Block GRU, online
replay, return normalization and slow critic are NOT included.
"""
import torch
from torch import nn
import torch.nn.functional as F

from world_model.categorical_rssm import CategoricalRSSM
from world_model.vision_losses import image_objective


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


def twohot(values, centers):
    """Interpolate each scalar between neighboring fixed bins."""
    values = values.clamp(centers[0], centers[-1])
    upper = torch.searchsorted(centers.contiguous(), values.contiguous()).clamp(1, len(centers)-1)
    lower = upper - 1
    low = centers[lower]
    high = centers[upper]
    w_upper = (values - low) / (high - low).clamp_min(1e-8)
    target = torch.zeros(*values.shape, len(centers), dtype=values.dtype, device=values.device)
    target.scatter_add_(-1, lower.unsqueeze(-1), (1-w_upper).unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), w_upper.unsqueeze(-1))
    return target


class V3RSSM(CategoricalRSSM):
    """Categorical RSSM + symlog/twohot scalar reward distribution."""

    def __init__(self, action_dim=3, hidden=256, groups=16, classes=16,
                 embed=256, unimix=0.01, reward_bins=255, reward_range=20.0):
        super().__init__(action_dim, hidden, groups, classes, embed, unimix)
        if reward_bins < 3 or reward_range <= 0:
            raise ValueError("Invalid reward bin configuration")
        self.reward_bins = reward_bins
        self.register_buffer("reward_centers",
                             torch.linspace(-reward_range, reward_range, reward_bins))
        self.reward_head = nn.Sequential(
            nn.Linear(hidden + groups * classes, hidden), nn.ELU(),
            nn.Linear(hidden, reward_bins))

    def decode(self, state):
        feat = self.features(state)
        logits = self.reward_head(feat)
        reward = symexp((logits.softmax(-1) * self.reward_centers).sum(-1))
        return self.decoder(feat), reward, self.continue_head(feat).squeeze(-1)

    def loss(self, observations, actions, rewards, dones, dyn_scale=0.05,
             rep_scale=0.01, free_nats=1.0, reward_scale=1.0,
             recon_mode="mse", ssim_weight=0.2, edge_weight=0.1):
        """Reward loss is two-hot cross entropy on symlog(raw reward), not L1."""
        if actions.shape[1] < 1:
            raise ValueError("Expected at least one transition")
        if reward_scale != 1.0:
            raise ValueError("V3 uses raw reward targets (reward_scale must be 1)")
        state = self.observe_first(observations[:, 0])
        sums = {key: observations.new_zeros(()) for key in
                ("recon_mse", "recon_objective", "reward_twohot_ce",
                 "continue_bce", "dyn_kl", "rep_kl")}
        for t in range(actions.shape[1]):
            state, prior, posterior = self.observe_step(
                state, actions[:, t], observations[:, t + 1])
            feat = self.features(state)
            frame = self.decoder(feat)
            logits = self.reward_head(feat)
            continuation = self.continue_head(feat).squeeze(-1)
            dyn, rep = self.balanced_kl(posterior, prior, free_nats)
            target = twohot(symlog(rewards[:, t]), self.reward_centers)
            reward_ce = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
            sums["recon_mse"] += F.mse_loss(frame, observations[:, t+1])
            sums["recon_objective"] += image_objective(
                frame, observations[:, t+1], recon_mode, ssim_weight, edge_weight)
            sums["reward_twohot_ce"] += reward_ce
            sums["continue_bce"] += F.binary_cross_entropy_with_logits(
                continuation, 1-dones[:, t])
            sums["dyn_kl"] += dyn
            sums["rep_kl"] += rep
        sums = {key: val / actions.shape[1] for key, val in sums.items()}
        sums["loss"] = (
            sums["recon_objective"] + sums["reward_twohot_ce"]
            + 0.1 * sums["continue_bce"]
            + dyn_scale * sums["dyn_kl"] + rep_scale * sums["rep_kl"])
        return sums
