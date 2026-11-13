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
