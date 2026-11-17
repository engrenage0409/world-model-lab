"""Data collection policy tests; Gymnasium environment not needed."""
import numpy as np
import pytest

from scripts.collect_diverse_data import new_action, collect
from argparse import Namespace


def test_smooth_and_random_action_ranges():
    rng = np.random.default_rng(123)
    last = None
    smooth = []
    for t in range(128):
        action = new_action(rng, "smooth", last, t)
        assert action.dtype == np.float32
        assert -1 <= action[0] <= 1
        assert 0 <= action[1] <= 1
        assert 0 <= action[2] <= 1
        smooth.append(action)
        last = action
    assert np.std([a[0] for a in smooth]) > 0
    for _ in range(10):
        action = new_action(rng, "random")
        assert action.shape == (3,)
    with pytest.raises(ValueError):
        new_action(rng, "unknown")


def test_collector_wont_overwrite(tmp_path):
    (tmp_path / "episode_0000.npz").write_bytes(b"important old data")
    args = Namespace(out=str(tmp_path), episodes=2, max_steps=3,
                     start_index=0, seed=42, policy="mixed")
    with pytest.raises(FileExistsError):
        collect(args)
    assert (tmp_path / "episode_0000.npz").read_bytes() == b"important old data"
