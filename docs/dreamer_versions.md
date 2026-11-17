# Dreamer V1 / V2 / V3: source-of-truth architecture tracks

Status updated 2026-11-17. These are **three distinct target architectures**,
not three labels for identical RSSM implementations.

| Track | Reference repository | Algorithm distinction | In world-model-lab |
|---|---|---|---|
| V1 | https://github.com/danijar/dreamer | Gaussian stochastic RSSM; imagined actor/critic using analytic value gradients | Gaussian RSSM world model implemented and locally trained; **actor/critic missing** |
| V2 | https://github.com/danijar/dreamerv2 | Categorical latent RSSM, straight-through gradients, KL balancing; policy learning by imagination | Categorical RSSM world model implemented and locally trained; **actor/critic missing** |
| V3 | https://github.com/danijar/dreamerv3 | Discrete RSSM with Unimix, dynamics/representation KL, symlog/two-hot, imagined actor/critic, return normalization | Separate **experimental V3-style RSSM** with symlog + two-hot reward, but **value two-hot, full actor/critic, Block GRU and online replay still missing** |

Do **not** claim these are exact reproductions: the current implementations
differ in network size, objective coefficients, training budgets, and missing
online agent components. The current categorical model is an exploratory
V2/V3 hybrid (discrete latent + Unimix + separate KL); it is **not** full V2.
The independent `V3RSSM` adds a distributional reward head, but is **not full DreamerV3**.

## External implementation choices

- **V1 official**: https://github.com/danijar/dreamer (TensorFlow 2, older dependency set).
- **V2 official**: https://github.com/danijar/dreamerv2 (TensorFlow 2, package available).
  The author's project page also links tested PyTorch reimplementations:
  https://github.com/jurgisp/pydreamer and
  https://github.com/jsikyoon/dreamer-torch.
- **V3 reference**: https://github.com/danijar/dreamerv3 (JAX, Linux/Mac-oriented).
  PyTorch community educational implementation:
  https://github.com/tesla-agi/Dreamer-v3 (Crafter-oriented and version-pinned).

Source-code downloads are available. **Compatible pretrained CarRacing-v3
checkpoints have not been verified**. Do not assume a Crafter or Atari checkpoint
can be loaded for our CarRacing model or that an official TensorFlow/JAX
checkpoint fits our PyTorch `.pt` format.

Do not install old TensorFlow, legacy Gym, or incompatible JAX dependencies
into the existing `pytorch_env`. Clone external reference repositories into
separate folders/environment(s) as needed. Adapt / rewrite using the original
license and attribution before integrating reused code.

## Unified evaluation contract (planned for full agents)

All versions should eventually expose:
- `observe(o_t, state)` to infer the latent state from a true observation
- `imagine(state, actions)` for closed-loop latent dynamics
- `decode(state)` for reconstruction and reward/continuation estimates
- `act(state)` for learned continuous CarRacing actions
- metrics: posterior frame MSE; prior open-loop frame MSE by horizon;
  reward MAE; policy episode return across unseen track seeds.

Current shared model evaluation covers the first three capabilities for
the Gaussian and discrete prototypes (no learned actor yet).

## Data experiment before next architecture changes

The original dataset: five random-action episodes, up to 300 steps each.
Use the independent collector for a 50-episode experiment:

```powershell
python -m scripts.collect_diverse_data --episodes 50 --max-steps 500 --policy mixed --out data/episodes_v2
python -m scripts.train_world_model --data data/episodes_v2 --epochs 30 --seq-len 8 --batch-size 2 --out checkpoints/gaussian_50ep_best.pt
python -m scripts.evaluate_world_model --data data/episodes_v2 --checkpoint checkpoints/gaussian_50ep_best.pt --out results/gaussian_50ep_eval.json
```

The collector uses independent environment seeds, temporally smooth control
for half of episodes and step-random control for the others, and writes a
`manifest.csv` with steps, reward, and action statistics. It does **not** implement
an expert/track-following driver, so many observations can still be off-road.
Do not present 50 episodes as a universal sufficient dataset size.
It refuses to overwrite existing episode files and defaults to a new folder.
To add more data later, choose an unused `--start-index` and retain the
same `--out` folder.

After evaluating 50 episodes, consider more diverse data (e.g. track-following
heuristics or trained policy exploration), sequence length 16/32, and
held-out evaluation on completely new track seeds. Only then implement
the complete V1/V2 imagined actor/critic and V3 robustness machinery.

## Three model variants in the current PyTorch project

- **V1-like baseline**: `RSSMWorldModel` in `world_model/rssm.py`, Gaussian stochastic latent.
  Train with `python -m scripts.train_world_model`.
- **V2-like discrete prototype**: `CategoricalRSSM` in
  `world_model/categorical_rssm.py`, straight-through categorical latent,
  balanced KL, and scalar reward Smooth L1.
  Train with `python -m scripts.train_categorical_world_model`.
- **V3-style experimental prototype**: `V3RSSM` in
  `world_model/v3_rssm.py`, categorical latent, Unimix, balanced KL,
  symlog + two-hot scalar reward distribution. It retains the simple
  CNN/GRU and no learned policy. This is not the complete DreamerV3.
  Train with `python -m scripts.train_v3_world_model`.

```powershell
python -m pytest -q
python -m scripts.train_v3_world_model --data data/episodes_v2 --epochs 30 --seq-len 8 --batch-size 2 --out checkpoints/v3_experimental_best.pt
python -m scripts.evaluate_world_model --data data/episodes_v2 --checkpoint checkpoints/v3_experimental_best.pt --out results/v3_experimental_eval.json
python -m scripts.compare_reconstruction --data data/episodes_v2 --checkpoint checkpoints/v3_experimental_best.pt --out results/v3_experimental.png
```

This V3-style checkpoint has `model_type=v3_experimental` and
`reward_multiplier=1` because its reward decoder returns raw reward
units, unlike the earlier scalar heads trained on reward/10.
Total training losses across these versions are not directly comparable,
particularly because two-hot reward CE has a different numerical scale.
Evaluate held-out open-loop image MSE, reward MAE and, eventually, episode
return under identical conditions instead.
