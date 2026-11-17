"""Learn the Dreamer-inspired encoder/RSSM/decoder/reward/continue models."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from world_model.data import SequenceDataset,split_episodes
from world_model.rssm import RSSMWorldModel

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/episodes")
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--seq-len",type=int,default=8)
    p.add_argument("--stride",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--kl-scale",type=float,default=0.01)
    p.add_argument("--recon-mode",choices=["mse","l1_ssim_edge"],default="mse")
    p.add_argument("--ssim-weight",type=float,default=0.2)
    p.add_argument("--edge-weight",type=float,default=0.1)
    p.add_argument("--out",default="checkpoints/rssm_best.pt")
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    train_files,val_files=split_episodes(a.data,seed=a.seed)
    train=DataLoader(SequenceDataset(train_files,a.seq_len,a.stride),
                     batch_size=a.batch_size,shuffle=True,num_workers=0)
    val=DataLoader(SequenceDataset(val_files,a.seq_len,a.stride),
                   batch_size=a.batch_size,shuffle=False,num_workers=0)
    model=RSSMWorldModel().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    best=float("inf")
    for epoch in range(1,a.epochs+1):
        summaries={}
        for split,loader in (("train",train),("val",val)):
            model.train(split=="train")
            metrics=[]
            for batch in tqdm(loader,desc=f"Epoch {epoch} {split}"):
                batch={k:v.to(device) for k,v in batch.items()}
                with torch.set_grad_enabled(split=="train"):
                    losses=model.loss(**batch,kl_scale=a.kl_scale,recon_mode=a.recon_mode,ssim_weight=a.ssim_weight,edge_weight=a.edge_weight)
                    loss=losses["loss"]
                    if split=="train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),100.)
                        optimizer.step()
                metrics.append({k:float(v.detach().item()) for k,v in losses.items()})
            summaries[split]={k:sum(v[k] for v in metrics)/len(metrics) for k in metrics[0]}
        print(json.dumps({"epoch":epoch,**summaries}))
        if summaries["val"]["loss"]<best:
            best=summaries["val"]["loss"]
            torch.save({"model":model.state_dict(),"config":{"hidden":256,"stoch":32,"embed":256},
                        "epoch":epoch,"recon_mode":a.recon_mode,"ssim_weight":a.ssim_weight,"edge_weight":a.edge_weight,"val":summaries["val"],"train_files":[x.name for x in train_files],
                        "val_files":[x.name for x in val_files],"seq_len":a.seq_len},out)
            print("Saved",out)
    print("Best validation loss:",best,"device:",device)

if __name__=="__main__":main()
