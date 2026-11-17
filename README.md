# World Model Lab — CarRacing RSSM → Dreamer → JEPA → 3D worlds

A staged world-model research repository on **CarRacing-v3**. The first implemented track is a **Dreamer-inspired Gaussian RSSM** with convolutional reconstruction, reward/continuation prediction and action-conditioned open-loop prediction.

**Status:** scripts and model are provided; **no verified trained model, quantitative result, or trained control policy is claimed yet**. This is not a reproduction of DreamerV3. Actor-Critic, JEPA and Atlas-style 3D reconstruction remain planned tracks.

## Architecture and temporal alignment

```text
o_t -- CNN encoder --> e_t
                           |
(z_t,h_t) + action a_t ---> GRU h_(t+1) --> Prior p(z_(t+1)|h_(t+1))
                                            |
o_(t+1) --> CNN encoder --> Posterior q(z_(t+1)|h_(t+1),e_(t+1))
                                            |
                                      (h_(t+1),z_(t+1))
                                      /        |       \
                             decoder      reward     continue
                             o_(t+1)      r_t        1-done_t

Training uses Posterior; open-loop prediction uses Prior given actions (not future frames).
```

Observations are RGB uint8 [T+1,96,96,3]; actions are [T,3] (steering, gas, brake). Reward and done contain T elements. This alignment is **intentional**: the action applied at time t generates the next observation and transition reward.

## Repository layout

```text
world_model/
  data.py                 .npz validation, episode-wise splitting, window dataset
  rssm.py                 CNN encoder/decoder, GRU + stochastic prior/posterior
scripts/
  collect_data.py         CarRacing-v3 random-policy data collector
  inspect_data.py         Show episode shapes and reward sums
  train_world_model.py    RSSM optimization (reconstruction, reward, continue, KL)
  evaluate_world_model.py Held-out open-loop frame MSE and reward MAE
  visualize_rollout.py    Ground truth vs open-loop imagined frames
tests/                    Dataset alignment and RSSM shape/gradient checks
docs/
  roadmap.md              Dreamer actor/critic, JEPA and Atlas milestones
  reading-list.md         Papers and official code links
```

## Installation (Windows PowerShell / RTX 5060 8GB)

Use your existing PyTorch conda environment. Install compatible CUDA PyTorch first if not already present; follow the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

```powershell
git clone https://github.com/engrenage0409/world-model-lab.git
cd world-model-lab
python -m pip install -r requirements.txt
python -m pytest -q
```

Gymnasium Box2D may require compiler/build dependencies on some Windows setups; if `gymnasium[box2d]` installation fails, check its upstream requirements.

## Step 1 — collect or reuse episode files

Your previously collected CarRacing episode files can be reused: copy the **five local .npz files** into `data/episodes/` (not committed to Git). To collect new ones:

```powershell
python -m scripts.collect_data --episodes 5 --max-steps 300
python -m scripts.inspect_data
```

Expected for an episode containing T steps:

```text
observations (T+1,96,96,3) uint8
actions      (T,3)        float32
rewards      (T,)         float32
dones        (T,)         bool
```

These scripts are independent from any earlier local `collect_data.py`: nothing overwrites the old files unless saved to the same output directory. Old files can be copied to `data/episodes` as-is if shape/dtype match.

## Step 2 — train RSSM

```powershell
python -m scripts.train_world_model --data data/episodes --epochs 5 --seq-len 8 --batch-size 2
```

Default 5 episode files => episode-level holdout (about 4 train / 1 validation). `data/`, `checkpoints/` and `results/` are ignored by Git. Model trains on sequence windows and produces `checkpoints/rssm_best.pt`.

The observation and reward models are deliberately lightweight: pixel MSE for reconstruction, reward Smooth L1 on reward/10, continuation BCE and Gaussian KL (clamped with 1 free nat per sequence step). They are **not DreamerV3's two-hot/symlog and categorical latent techniques**. With only 5 random-policy episodes, expect this to be a pipeline learning exercise, not a capable driving agent.

## Step 3 — evaluate and visualize

```powershell
python -m scripts.evaluate_world_model
python -m scripts.visualize_rollout
```

Evaluation reports held-out **action-conditioned open-loop** image MSE and reward MAE. The visualization writes `results/rollout.png` with ground truth vs imagined prior frames. The model gets the **true actions**, not subsequent observations after t=0; it does not autonomously choose actions.

## Next development

See [detailed roadmap](docs/roadmap.md) for the actor/critic + imagined λ-return phase, DreamerV2/V3 improvements, JEPA / action-conditioned JEPA and Atlas-inspired 3DGS reconstruction. See [reading list](docs/reading-list.md) for original papers.

**Reproducibility:** fix episode splits, seeds and library versions, collect more diverse episodes, log per-episode returns, and evaluate on fresh held-out trajectories. This repository is educational research code, not production autonomous driving.

## Experimental categorical RSSM (DreamerV3-inspired, not full DreamerV3)

The existing Gaussian RSSM and its checkpoint are unchanged. A second model,
`world_model/categorical_rssm.py`, has discrete one-hot stochastic state
(`groups=16`, `classes=16`) with 1% uniform probability mixing, a straight-through
gradient estimator, and separately stop-gradiented Dynamics and Representation
KL losses with per-sample free nats. It uses the same CNN reconstruction,
reward Smooth L1, and continuation BCE heads as the Gaussian baseline.

To compare, first make sure the old model and five episode files are present.
The categorical model trains **from scratch**; no Gaussian weights are reused.

```powershell
git pull origin main
python -m pytest -q
python -m scripts.train_categorical_world_model --data data/episodes --epochs 30 --seq-len 8 --batch-size 2 --out checkpoints/categorical_best.pt
python -m scripts.evaluate_world_model --checkpoint checkpoints/categorical_best.pt --out results/categorical_eval.json
python -m scripts.compare_reconstruction --checkpoint checkpoints/categorical_best.pt --out results/categorical_comparison.png
```

The original Gaussian checkpoint can still be assessed with
`python -m scripts.evaluate_world_model --checkpoint checkpoints/rssm_best.pt`.
The checkpoint stores `model_type` and configuration, and the shared evaluation
and visualization scripts detect the correct model.

**Fair-comparison notes:** hold the episode split, seed, sequence length, training
budget, action data and metrics fixed. A 30-epoch run saves the *lowest validation
loss* checkpoint, not necessarily epoch 30. The default KL weights (`--dyn-scale
0.05 --rep-scale 0.01`) are deliberately small for the existing mean-pixel-MSE
scale; they are **not** official DreamerV3 loss coefficients. This prototype
does not include Block GRU, symlog/two-hot, policy learning or online replay.

## Vision reconstruction ablation (experimental)

Both Gaussian and Categorical RSSMs now accept `--recon-mode mse`
(default, backward compatible) or `--recon-mode l1_ssim_edge` with
`--ssim-weight 0.2 --edge-weight 0.1`. The latter trains with
L1 + a lightweight local 3x3 SSIM surrogate + image-gradient L1.
This is **not** DreamerV3's official visual loss. Training logs
`recon_mse` for comparability and `recon_objective` for the optimized
reconstruction loss. Comparing total losses across modes is invalid;
compare held-out image metrics and visualizations instead.

To compare the Gaussian baseline under the *same data and seed*:

```powershell
python -m pytest -q
python -m scripts.evaluate_world_model --checkpoint checkpoints/rssm_30ep.pt --out results/gaussian_30ep_eval.json
python -m scripts.train_world_model --data data/episodes --epochs 30 --seq-len 8 --batch-size 2 --recon-mode l1_ssim_edge --out checkpoints/gaussian_vision_best.pt
python -m scripts.evaluate_world_model --checkpoint checkpoints/gaussian_vision_best.pt --out results/gaussian_vision_eval.json
python -m scripts.compare_reconstruction --checkpoint checkpoints/gaussian_vision_best.pt --out results/gaussian_vision.png
```

`evaluate_world_model` now also writes `open_loop_frame_mse_by_step`,
a vector of 1..T-step errors averaged over validation windows. The
validation split is at the episode level. Longer-horizon prediction and
more diverse trajectories are still future research steps; no claim is
made that the new loss necessarily improves generalization.

## Next experiments: data scale and three Dreamer architecture tracks

The immediate bottleneck may be only five short random trajectories. A new
non-overwriting, manifest-producing collector supports three policies
(`random`, `smooth`, `mixed`) and runs separately from the original collector:

```powershell
python -m scripts.collect_diverse_data --episodes 50 --max-steps 500 --policy mixed --out data/episodes_v2
python -m scripts.train_world_model --data data/episodes_v2 --epochs 30 --seq-len 8 --batch-size 2 --out checkpoints/gaussian_50ep_best.pt
python -m scripts.evaluate_world_model --data data/episodes_v2 --checkpoint checkpoints/gaussian_50ep_best.pt --out results/gaussian_50ep_eval.json
```

These policies are exploratory, not expert drivers. Larger data alone does not
guarantee accurate on-road dynamics. Old `data/episodes` and `checkpoints/`
are unchanged.

See [DreamerV1/V2/V3 architecture and open-source reuse plan](docs/dreamer_versions.md).
The current Gaussian model corresponds to an **incomplete V1 world-model
baseline**, the categorical model is a **V2/V3-inspired hybrid**, and a full
V3 training/actor/critic implementation is **not yet present**. Each version
should eventually get its own architecture, independent checkpoint format, and
a shared evaluation on common CarRacing trajectories. Do not claim a full
DreamerV1/V2/V3 reproduction from these baselines.
