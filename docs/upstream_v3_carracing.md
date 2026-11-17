# Upstream DreamerV3 → CarRacing-v3 integration status

## Why source checkout alone is insufficient

Verified against `burchim/DreamerV3-PyTorch` at
`a6ee1d86cf3e387694d58a49aa3e81a5b010d8c5`:

- `main.py` imports `configs/DreamerV3/dreamer_v3.py`, which
  requires an `env_name` environment variable and constructs a real
  `nnet.models.rl.dreamer.DreamerV3` agent.
- `nnet/models/rl/dreamer/dreamerv3.py` accepts environment prefixes
  `dmc`, `atari`, `atari100k`, `minerl`, `memory_maze`; there is no
  CarRacing branch.
- Its DMC wrapper uses torch tensors in `state`, `reward`, `done`,
  `is_first`, `is_last`, and exposes `num_actions`, `action_size`,
  `action_repeat`. This is NOT the standard Gymnasium tuple interface.
- Upstream dependencies include legacy Gym 0.19. Installing its full
  requirements file into our modern `pytorch_env` is not supported.

## First reproducible diagnostic

```powershell
git pull origin main
python -m scripts.check_upstream_env --steps 5
```

This checks all pinned upstream Git revisions and validates live
CarRacing-v3 observation/action shapes via Gymnasium. It does not
import or train upstream DreamerV3, and does not require installing
legacy Gym packages.

## Adaptation plan

1. **Environment adapter:** Add CarRacing support to an isolated fork/
   patch of upstream V3. Gymnasium emits `reset -> (obs, info)`, and
   `step -> (obs, reward, terminated, truncated, info)`; the model
   needs `state/reward/done/is_first/is_last` tensors. Clip/transform
   normalized actor actions to `steer [-1,1]`, `gas [0,1]`,
   `brake [0,1]`. Preserve terminal vs truncation semantics.
2. **Config selection:** Add CarRacing to the `env_type` configuration
   and its many lookup tables. Use small model size and batch size
   initially for an 8GB laptop GPU.
3. **Dependency isolation:** Test a clean isolated environment with
   compatible torchvision and required upstream modules. Prefer WSL2
   if legacy dependency imports fail on Windows; do not pin outdated
   Gym in the currently working conda environment.
4. **Model + replay smoke test:** Construct the original upstream
   `DreamerV3`, collect a short episode, sample replay, run one
   actor/critic/model update, validate tensors and checkpoint resume.
5. **Real training:** Expand on-road data or online interaction and
   track returns and imaging metrics. Do not compare against V1/V2 until
   the tasks and evaluation protocol match.

No upstream CarRacing model has yet been trained or validated; code
download and the environment diagnostic are separate milestones.
