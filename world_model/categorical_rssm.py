"""DreamerV3-inspired categorical RSSM; does not implement full DreamerV3."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import OneHotCategoricalStraightThrough
import torch.nn.functional as F

from world_model.rssm import CNNEncoder, CNNDecoder
from world_model.vision_losses import image_objective


@dataclass
class CategoricalState:
    h: torch.Tensor
    z: torch.Tensor  # [batch, groups, classes]


class CategoricalRSSM(nn.Module):
    def __init__(self, action_dim=3, hidden=256, groups=16, classes=16,
                 embed=256, unimix=0.01):
        super().__init__()
        if groups < 1 or classes < 2 or not 0 <= unimix < 1:
            raise ValueError("Invalid categorical latent configuration")
        self.hidden, self.groups, self.classes = hidden, groups, classes
        self.unimix = unimix
        self.encoder = CNNEncoder(embed)
        latent = groups * classes
        self.gru = nn.GRUCell(latent + action_dim, hidden)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, latent))
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden + embed, hidden), nn.ELU(),
            nn.Linear(hidden, latent))
        feature_dim = hidden + latent
        self.decoder = CNNDecoder(feature_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.continue_head = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ELU(), nn.Linear(hidden, 1))

    def init_state(self, batch, device):
        return CategoricalState(
            torch.zeros(batch, self.hidden, device=device),
            torch.zeros(batch, self.groups, self.classes, device=device))

    def distribution(self, raw_logits):
        logits = raw_logits.reshape(-1, self.groups, self.classes)
        probs = F.softmax(logits, dim=-1)
        probs = (1 - self.unimix) * probs + self.unimix / self.classes
        return OneHotCategoricalStraightThrough(probs=probs)

    @staticmethod
    def features(state):
        return torch.cat([state.h, state.z.flatten(1)], dim=-1)

    def observe_first(self, image):
        h = self.init_state(image.shape[0], image.device).h
        encoded = self.encoder(image)
        posterior = self.distribution(self.posterior_net(torch.cat([h, encoded], -1)))
        return CategoricalState(h, posterior.rsample())

    def prior_step(self, state, action, sample=True):
        h = self.gru(torch.cat([state.z.flatten(1), action], -1), state.h)
        prior = self.distribution(self.prior_net(h))
        z = prior.rsample() if sample else F.one_hot(
            prior.probs.argmax(-1), self.classes).to(h.dtype)
        return CategoricalState(h, z), prior

    def observe_step(self, state, action, next_image):
        prediction, prior = self.prior_step(state, action)
        encoded = self.encoder(next_image)
        posterior = self.distribution(
            self.posterior_net(torch.cat([prediction.h, encoded], -1)))
        return CategoricalState(prediction.h, posterior.rsample()), prior, posterior

    def decode(self, state):
        features = self.features(state)
        return (self.decoder(features), self.reward_head(features).squeeze(-1),
                self.continue_head(features).squeeze(-1))

    @staticmethod
    def balanced_kl(posterior, prior, free_nats=1.0):
        """Stop-gradient KL(q.detach || p) and KL(q || p.detach)."""
        q, p = posterior.probs, prior.probs
        eps = 1e-8
        dynamics = (q.detach() * (
            (q.detach() + eps).log() - (p + eps).log())).sum((-1, -2))
        representation = (q * (
            (q + eps).log() - (p.detach() + eps).log())).sum((-1, -2))
        return (dynamics.clamp_min(free_nats).mean(),
                representation.clamp_min(free_nats).mean())

    def loss(self, observations, actions, rewards, dones, dyn_scale=0.05,
             rep_scale=0.01, free_nats=1.0, reward_scale=10.0,
             recon_mode="mse", ssim_weight=0.2, edge_weight=0.1):
        state = self.observe_first(observations[:, 0])
        sums = {key: observations.new_zeros(()) for key in
                ("recon_mse", "recon_objective", "reward_smooth_l1", "continue_bce", "dyn_kl", "rep_kl")}
        steps = actions.shape[1]
        if steps < 1:
            raise ValueError("Sequence must contain at least one action")
        for t in range(steps):
            state, prior, posterior = self.observe_step(
                state, actions[:, t], observations[:, t + 1])
            frame, reward, continuation = self.decode(state)
            dyn, rep = self.balanced_kl(posterior, prior, free_nats)
            sums["recon_mse"] = sums["recon_mse"] + F.mse_loss(
                frame, observations[:, t + 1])
            sums["recon_objective"] = sums["recon_objective"] + image_objective(
                frame, observations[:, t + 1], recon_mode, ssim_weight, edge_weight)
            sums["reward_smooth_l1"] = sums["reward_smooth_l1"] + F.smooth_l1_loss(
                reward, rewards[:, t] / reward_scale)
            sums["continue_bce"] = sums["continue_bce"] + F.binary_cross_entropy_with_logits(
                continuation, 1 - dones[:, t])
            sums["dyn_kl"] = sums["dyn_kl"] + dyn
            sums["rep_kl"] = sums["rep_kl"] + rep
        sums = {key: value / steps for key, value in sums.items()}
        sums["loss"] = (
            sums["recon_objective"] + sums["reward_smooth_l1"] +
            0.1 * sums["continue_bce"] +
            dyn_scale * sums["dyn_kl"] + rep_scale * sums["rep_kl"])
        return sums

    @torch.no_grad()
    def predict_open_loop(self, start_image, actions):
        self.eval()
        state = self.observe_first(start_image)
        frames, rewards = [], []
        for t in range(actions.shape[1]):
            state, _ = self.prior_step(state, actions[:, t], sample=False)
            frame, reward, _ = self.decode(state)
            frames.append(frame)
            rewards.append(reward)
        return torch.stack(frames, 1), torch.stack(rewards, 1)
