import numpy as np
import pytest
from world_model.data import load_episode,SequenceDataset,split_episodes

def test_episode_alignment(tmp_path):
    for i in range(3):
        np.savez_compressed(tmp_path/f"e{i}.npz",
            observations=np.zeros((13,96,96,3),dtype=np.uint8),
            actions=np.zeros((12,3),dtype=np.float32),
            rewards=np.zeros(12,dtype=np.float32),
            dones=np.array([False]*11+[True]))
    train,val=split_episodes(tmp_path,seed=42)
    assert len(train)==2 and len(val)==1
    sample=SequenceDataset(train,length=8,stride=4)[0]
    assert sample["observations"].shape==(9,3,96,96)
    assert sample["actions"].shape==(8,3)
    assert sample["rewards"].shape==(8,)

def test_bad_episode(tmp_path):
    path=tmp_path/"bad.npz"
    np.savez_compressed(path,observations=np.zeros((4,96,96,3),dtype=np.uint8),
        actions=np.zeros((4,3)),rewards=np.zeros(4),dones=np.zeros(4))
    with pytest.raises(ValueError): load_episode(path)
