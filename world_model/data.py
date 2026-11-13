"""CarRacing episode arrays: observations[T+1,H,W,3], actions[T,3], rewards[T], dones[T]."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def load_episode(path):
    with np.load(path,allow_pickle=False) as f:
        needed=("observations","actions","rewards","dones")
        missing=[k for k in needed if k not in f]
        if missing: raise ValueError(f"{path}: missing keys {missing}")
        obs,actions,rewards,dones=(f[k].copy() for k in needed)
    t=len(actions)
    if len(obs)!=t+1 or len(rewards)!=t or len(dones)!=t:
        raise ValueError(f"{path}: expected T+1 observations and T action/reward/done entries")
    if obs.ndim!=4 or obs.shape[1:]!=(96,96,3) or obs.dtype!=np.uint8:
        raise ValueError(f"{path}: need uint8 [T+1,96,96,3] observations")
    if actions.shape!=(t,3):
        raise ValueError(f"{path}: expected actions [T,3]")
    if not np.isfinite(actions).all() or not np.isfinite(rewards).all():
        raise ValueError(f"{path}: non-finite actions or rewards")
    return obs,actions.astype("float32"),rewards.astype("float32"),dones.astype("bool")

def split_episodes(folder,val_fraction=0.2,seed=42):
    paths=sorted(Path(folder).glob("*.npz"))
    if len(paths)<2: raise ValueError("Need >=2 episode files to make episode-level train/val split")
    rng=np.random.default_rng(seed)
    indices=rng.permutation(len(paths))
    nval=max(1,min(len(paths)-1,round(len(paths)*val_fraction)))
    val_indices=set(indices[:nval])
    return [p for i,p in enumerate(paths) if i not in val_indices],[p for i,p in enumerate(paths) if i in val_indices]

class SequenceDataset(Dataset):
    """Sliding windows per episode; never crosses an episode boundary."""
    def __init__(self,paths,length=16,stride=8):
        if length<1 or stride<1: raise ValueError("length and stride must be positive")
        self.episodes=[load_episode(p) for p in paths]
        self.windows=[]
        for i,(_,actions,_,_) in enumerate(self.episodes):
            for start in range(0,len(actions)-length+1,stride):
                # If a saved episode contains internal terminals, refuse to cross them.
                done=self.episodes[i][3]
                if done[start:start+length-1].any(): continue
                self.windows.append((i,start))
        if not self.windows: raise ValueError("No windows found; reduce --seq-len")
        self.length=length
    def __len__(self): return len(self.windows)
    def __getitem__(self,index):
        ep,start=self.windows[index]
        obs,actions,rewards,dones=self.episodes[ep]
        length=self.length
        frames=np.ascontiguousarray(obs[start:start+length+1].transpose(0,3,1,2))
        return {
            "observations":torch.from_numpy(frames).float().div(255.0),
            "actions":torch.from_numpy(actions[start:start+length].copy()),
            "rewards":torch.from_numpy(rewards[start:start+length].copy()),
            "dones":torch.from_numpy(dones[start:start+length].astype("float32").copy())
        }
