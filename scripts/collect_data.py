"""Collect CarRacing-v3 transitions in the exact T+1 observation / T action format."""
import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np

def policy(obs,rng):
    # Exploratory random actions; not a trained driver.
    return np.array([rng.uniform(-0.7,0.7),rng.uniform(0.1,0.8),0.0],dtype=np.float32)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--episodes",type=int,default=5)
    p.add_argument("--max-steps",type=int,default=300)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--out",default="data/episodes")
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed)
    env=gym.make("CarRacing-v3",render_mode="rgb_array",continuous=True)
    try:
        for episode in range(a.episodes):
            obs,_=env.reset(seed=a.seed+episode)
            observations=[obs];actions=[];rewards=[];dones=[]
            for t in range(a.max_steps):
                action=policy(obs,rng)
                next_obs,reward,terminated,truncated,_=env.step(action)
                done=terminated or truncated
                observations.append(next_obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                obs=next_obs
                if done: break
            filename=out/f"episode_{episode:04d}.npz"
            np.savez_compressed(filename,
                observations=np.asarray(observations,dtype=np.uint8),
                actions=np.asarray(actions,dtype=np.float32),
                rewards=np.asarray(rewards,dtype=np.float32),
                dones=np.asarray(dones,dtype=bool))
            print(f"{filename}: steps={len(actions)} reward_sum={sum(rewards):.2f}")
    finally:
        env.close()

if __name__=="__main__":main()
