import argparse
from pathlib import Path
from world_model.data import load_episode

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/episodes")
    a=p.parse_args()
    for path in sorted(Path(a.data).glob("*.npz")):
        obs,act,rew,done=load_episode(path)
        print(path.name,"observations",obs.shape,"actions",act.shape,
              "rewards",rew.shape,"dones",done.shape,
              "total_reward",round(float(rew.sum()),2))
if __name__=="__main__":main()
