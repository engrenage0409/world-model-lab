# Reusing the real Dreamer V1 / V2 / V3 implementations

This directory does **not** contain rewritten copies of the upstream algorithms.
It contains a **pinned source manifest** at [sources.json](sources.json) and a
reproducible installer at [../scripts/setup_baselines.py](../scripts/setup_baselines.py).
The upstream Git repositories are downloaded on demand to `external/` and
ignored by Git, preserving license files and commit history.

## What is sourced from upstream?

| Alias | Upstream | Commit pinned | Framework | Supported as-is |
|---|---|---|---|---|
| V1 | https://github.com/danijar/dreamer | `56d4d444dfd0` | Legacy TensorFlow 2.2 | Official DMC example; **not** CarRacing-v3 |
| V2 | https://github.com/jurgisp/pydreamer | `8a4ea85a925b` | PyTorch, legacy Gym 0.19 | Atari/DMC/etc.; CarRacing adapter needed |
| V3 | https://github.com/burchim/DreamerV3-PyTorch | `a6ee1d86cf3e` | PyTorch, legacy Gym 0.19 | DMC/Atari training; CarRacing adapter needed |

The pinned revisions are available from the referenced public GitHub repositories.
Their code and licenses remain owned by their original maintainers:
V1/V2 are MIT-licensed; V3 PyTorch is Apache-2.0. Cite their original work in
papers and README or reports.

## Download independently on Windows PowerShell

From our project root, with Git installed:

```powershell
python -m scripts.setup_baselines --version all --dry-run
python -m scripts.setup_baselines --version v3
```

To download all three:

```powershell
python -m scripts.setup_baselines --version all
```

The script clones pinned revisions under:
```text
external/dreamer_v1/
external/dreamer_v2/
external/dreamer_v3/
```

It refuses to overwrite an existing checkout, and never installs
third-party requirements into `pytorch_env`. To inspect a downloaded source:

```powershell
git -C external/dreamer_v3 rev-parse HEAD
```

This must match the V3 commit SHA in `baselines/sources.json`.

## IMPORTANT: current integration status

- **Done:** selected and verified original repositories, licenses, Git revisions;
  created a reproducible multi-source checkout mechanism and separated their
  code from our educational RSSM prototypes.
- **Not done:** actual local download until the setup command is run on your
  machine; installing and validating external dependencies; CarRacing-v3
  environment wrapper; converting `data/episodes/*.npz` into each upstream
  replay format; training or evaluating the external agents on CarRacing.
- **No compatible pretrained CarRacing-v3 weights have been verified.**
  Downloading source code alone will not create an agent that can drive the car.
- **Risk:** V2/V3 reference repos depend on old Gym 0.19 and potentially old
  OS-specific dependencies. Do not run `pip install -r requirements.txt`
  inside your currently working `pytorch_env`. Use fresh virtual environments
  or WSL2/Linux, and validate modern GPU/CUDA availability separately.
- This work does **not** modify or replace the Gaussian, Categorical or
  experimental V3-style local RSSM baselines or any local checkpoints.

## Roadmap for genuinely training the upstream agents on CarRacing-v3

1. Start with V3 PyTorch (preferred) in an isolated environment; verify import,
   configuration and smoke-test inference with synthetic states.
2. Implement a Gymnasium adapter supplying 96x96 RGB, continuous actions
   [steering, gas, brake], correct `terminated` / `truncated` handling,
   and consistent reward/continuation semantics.
3. Adapt its rollout collector/replay storage, choosing either online
   interactions or the required conversion from local NPZ transitions.
4. Run a small CPU/GPU smoke test, then scale up to diverse on-road data and
   full actor-critic training. Only compare predictions/returns on shared
   environment seeds after this validation.
5. Add V2 and V1 in independent environments, preserving their original
   algorithms; never call our simplified PyTorch prototypes exact reproductions.

The original repositories also provide their own default tasks; those can be
useful quick implementation checks but are **not** a CarRacing benchmark.
