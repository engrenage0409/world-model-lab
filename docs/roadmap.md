# Staged implementation roadmap

### Phase 0 — environment and sequence integrity
- [x] CarRacing-v3 collector; T+1 observations and T actions/rewards/dones
- [x] Episode-level train/validation split (no frame leakage)
- [x] Sliding-window dataset with T+1/T temporal alignment

### Phase 1 — Gaussian RSSM world model (implemented; not yet trained in user's environment)
- [x] Conv encoder/decoder; GRU deterministic state h; Gaussian stochastic z
- [x] Prior p(z|h); posterior q(z|h,encoded observation)
- [x] Observation MSE, reward Smooth L1, continue BCE, KL terms
- [x] Training and held-out open-loop evaluation scripts
- [x] Frame-by-frame rollout visualization
- [ ] User-run training results; inspect posterior collapse/poor predicted frames

### Phase 2 — Dreamer-inspired behavior learning (not implemented)
- [ ] Reward and continuation calibrated under prior rollout
- [ ] Imagination rollout with learned actor actions and critic value estimates
- [ ] Lambda returns, discount weighting, actor loss / critic targets
- [ ] Replay buffer, policy interaction and periodic online data collection
- [ ] Closed-loop CarRacing benchmark vs random/heuristic agents

### Phase 3 — stronger research (not implemented)
- [ ] Discrete stochastic latents (DreamerV2/V3)
- [ ] KL balancing, symlog/two-hot heads, free bits calibration
- [ ] Ablations: latent sizes, KL scale, rollout horizon and data size
- [ ] Multiple seeds; PSNR/SSIM and task return, uncertainty analysis

### Later tracks (planned)
- V-JEPA / Action-conditioned JEPA
- 3D persistent world / Atlas-inspired depth+3DGS
- Integrated benchmark comparisons

Note: 5 episodes / ~1K transitions are adequate for debugging but generally insufficient for strong world modeling or policy learning. Track training/validation by episode and expand data.
