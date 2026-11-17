"""Held-out episode evaluation. Reconstruct observed next frames and open-loop prior frames."""
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from world_model.data import SequenceDataset
from world_model.checkpoints import model_from_checkpoint

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/episodes")
    p.add_argument("--checkpoint",default="checkpoints/rssm_best.pt")
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--out",default="results/world_model_eval.json")
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    model=model_from_checkpoint(ckpt).to(device)
    model.eval()
    files=[Path(a.data)/x for x in ckpt["val_files"]]
    val=DataLoader(SequenceDataset(files,ckpt["seq_len"],ckpt["seq_len"]),
                   batch_size=a.batch_size,shuffle=False,num_workers=0)
    total=0;prior_sum=0.;rew_sum=0.;num_frames=0
    step_sse=None;step_count=0
    with torch.inference_mode():
        for batch in val:
            batch={k:v.to(device) for k,v in batch.items()}
            predicted,rewards=model.predict_open_loop(batch["observations"][:,0],batch["actions"])
            target=batch["observations"][:,1:]
            prior_sum+=(predicted-target).square().sum().item()
            batch_step_mse=(predicted-target).square().mean(dim=(0,2,3,4))
            if step_sse is None: step_sse=torch.zeros_like(batch_step_mse)
            step_sse+=batch_step_mse*target.shape[0]
            step_count+=target.shape[0]
            rew_sum+=(rewards*10-batch["rewards"]).abs().sum().item()
            num_frames+=target.numel()
            total+=batch["rewards"].numel()
    result={"val_episodes":ckpt["val_files"],"windows":len(val.dataset),
            "open_loop_frame_mse":prior_sum/num_frames,
            "open_loop_reward_mae":rew_sum/total,
            "open_loop_frame_mse_by_step":(step_sse/step_count).cpu().tolist(),
            "caveat":"Open-loop predictions conditioned on ground-truth future actions; not policy control."}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
