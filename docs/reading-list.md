# World Model research references

## Required: RSSM and Dreamer
- [PlaNet: Learning Latent Dynamics for Planning from Pixels (2019)](https://arxiv.org/abs/1811.04551) — latent state planning, RSSM foundations.
- [Dream to Control: Learning Behaviors by Latent Imagination (DreamerV1)](https://arxiv.org/abs/1912.01603) — latent imagination and actor–critic.
- [Mastering Atari with Discrete World Models (DreamerV2)](https://arxiv.org/abs/2010.02193) — discrete stochastic latent state and actor–critic.
- [Mastering Diverse Domains through World Models (DreamerV3)](https://arxiv.org/abs/2301.04104) — unified generalist learning across domains.
- [DreamerV3 official code](https://github.com/danijar/dreamerv3) — reference for full architecture, distributional heads, symlog/symexp, KL balancing, and behavior learning.

## Next stages: representation and action
- [I-JEPA](https://arxiv.org/abs/2301.08243) — predictive representations without pixel decoding.
- [V-JEPA](https://arxiv.org/abs/2404.08471) — masked video representation prediction.
- [V-JEPA 2](https://arxiv.org/abs/2506.09985) — video and action-conditioned physical prediction.
- [TD-MPC2](https://arxiv.org/abs/2310.16828) — latent-model predictive control alternative.
- [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) — interactive world generation.

## 3D persistent world / Atlas-inspired
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — explicit 3D representation and rendering.
- [Nerfstudio](https://docs.nerf.studio/) — practical NeRF/3DGS reconstruction and camera geometry.
- Future Atlas-style track: depth, camera pose estimation, explicit scene reconstruction, novel view rendering and unseen-region completion. **No Atlas model reproduction is claimed.**

## Experiment interpretation
- Current checkpoint trains a Gaussian RSSM world model, not DreamerV3.
- Posterior uses observed next frame; prior predicts next state without the future frame.
- KL regularizes posterior towards prior. Reward prediction targets transition r_t; decoder reconstructs o_(t+1).
- A fixed action-conditioned open-loop rollout is *not* an autonomous agent; a learned actor/critic and imagination rollout training are separate future phases.
