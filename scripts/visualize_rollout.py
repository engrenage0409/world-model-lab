"""Save strips: ground-truth future frame vs open-loop prior prediction per action."""
import argparse
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from world_model.data import SequenceDataset
from world_model.checkpoints import model_from_checkpoint

def img(t):
    return Image.fromarray((t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype("uint8"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/episodes")
    p.add_argument("--checkpoint",default="checkpoints/rssm_best.pt")
    p.add_argument("--out",default="results/rollout.png")
    p.add_argument("--index",type=int,default=0)
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    model=model_from_checkpoint(ckpt).to(device)
    model.eval()
    paths=[Path(a.data)/name for name in ckpt["val_files"]]
    sample=SequenceDataset(paths,ckpt["seq_len"],ckpt["seq_len"])[a.index]
    with torch.inference_mode():
        pred,_=model.predict_open_loop(sample["observations"][None,0].to(device),
                                        sample["actions"][None].to(device))
    n=pred.shape[1]
    canvas=Image.new("RGB",(96*n,220),"white")
    draw=ImageDraw.Draw(canvas)
    for i in range(n):
        canvas.paste(img(sample["observations"][i+1]),(96*i,14))
        canvas.paste(img(pred[0,i]),(96*i,116))
    draw.text((2,0),"Ground truth o_(t+1)",fill="black")
    draw.text((2,102),"RSSM prior predicted (given action)",fill="black")
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(out)
    print("Saved",out)

if __name__=="__main__":main()
