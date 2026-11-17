"""Gaussian RSSM: prior p(z_t|h_t), posterior q(z_t|h_t,e_t), GRU dynamics.
Time: posterior z_0 from observation o_0. a_t takes (h_t,z_t)->h_{t+1}.
Posterior on o_{t+1}; decode o_{t+1}, predict transition reward r_t.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Normal,kl_divergence
import torch.nn.functional as F
from world_model.vision_losses import image_objective

class CNNEncoder(nn.Module):
    def __init__(self,embed_dim=256):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,4,2,1),nn.ELU(),
            nn.Conv2d(32,64,4,2,1),nn.ELU(),
            nn.Conv2d(64,128,4,2,1),nn.ELU(),
            nn.Conv2d(128,128,4,2,1),nn.ELU(),
            nn.Flatten(),nn.Linear(128*6*6,embed_dim),nn.LayerNorm(embed_dim))
    def forward(self,x): return self.net(x)

class CNNDecoder(nn.Module):
    def __init__(self,state_dim):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(state_dim,128*6*6),nn.ELU())
        self.net=nn.Sequential(
            nn.ConvTranspose2d(128,128,4,2,1),nn.ELU(),
            nn.ConvTranspose2d(128,64,4,2,1),nn.ELU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.ELU(),
            nn.ConvTranspose2d(32,3,4,2,1),nn.Sigmoid())
    def forward(self,s):
        return self.net(self.project(s).reshape(-1,128,6,6))

@dataclass
class RSSMState:
    h: torch.Tensor
    z: torch.Tensor

class RSSMWorldModel(nn.Module):
    def __init__(self,action_dim=3,hidden=256,stoch=32,embed=256):
        super().__init__()
        self.hidden,self.stoch=hidden,stoch
        self.encoder=CNNEncoder(embed)
        self.gru=nn.GRUCell(stoch+action_dim,hidden)
        self.prior_net=nn.Sequential(nn.Linear(hidden,hidden),nn.ELU(),nn.Linear(hidden,2*stoch))
        self.posterior_net=nn.Sequential(nn.Linear(hidden+embed,hidden),nn.ELU(),nn.Linear(hidden,2*stoch))
        state_dim=hidden+stoch
        self.decoder=CNNDecoder(state_dim)
        self.reward_head=nn.Sequential(nn.Linear(state_dim,hidden),nn.ELU(),nn.Linear(hidden,1))
        self.continue_head=nn.Sequential(nn.Linear(state_dim,hidden),nn.ELU(),nn.Linear(hidden,1))
    def init_state(self,batch,device):
        return RSSMState(torch.zeros(batch,self.hidden,device=device),
                         torch.zeros(batch,self.stoch,device=device))
    @staticmethod
    def normal(params):
        mean,raw_std=params.chunk(2,-1)
        std=F.softplus(raw_std)+0.1
        return Normal(mean,std)
    @staticmethod
    def features(state):
        return torch.cat([state.h,state.z],dim=-1)
    def observe_first(self,image):
        h=self.init_state(image.shape[0],image.device).h
        posterior=self.normal(self.posterior_net(torch.cat([h,self.encoder(image)],dim=-1)))
        return RSSMState(h,posterior.rsample())
    def prior_step(self,state,action,sample=True):
        h=self.gru(torch.cat([state.z,action],dim=-1),state.h)
        prior=self.normal(self.prior_net(h))
        z=prior.rsample() if sample else prior.mean
        return RSSMState(h,z),prior
    def observe_step(self,state,action,next_image):
        pred,prior=self.prior_step(state,action)
        posterior=self.normal(self.posterior_net(torch.cat([pred.h,self.encoder(next_image)],dim=-1)))
        return RSSMState(pred.h,posterior.rsample()),prior,posterior
    def decode(self,state):
        s=self.features(state)
        return self.decoder(s),self.reward_head(s).squeeze(-1),self.continue_head(s).squeeze(-1)
    def loss(self,observations,actions,rewards,dones,kl_scale=0.01,free_nats=1.0,reward_scale=10.0,recon_mode="mse",ssim_weight=0.2,edge_weight=0.1):
        """Aligned arrays: observations[B,T+1,3,96,96], actions/rewards/dones[B,T,...].
        Decoder reconstructs o_(t+1), reward predicts r_t, continue predicts 1-done_t.
        """
        state=self.observe_first(observations[:,0])
        total_rec,total_rew,total_cont,total_kl=0.,0.,0.,0.
        total_pixel_mse=0.
        steps=actions.shape[1]
        for t in range(steps):
            state,prior,posterior=self.observe_step(state,actions[:,t],observations[:,t+1])
            reconstructed,pred_reward,pred_continue=self.decode(state)
            rec=image_objective(reconstructed,observations[:,t+1],recon_mode,ssim_weight,edge_weight)
            total_pixel_mse=total_pixel_mse+F.mse_loss(reconstructed,observations[:,t+1])
            rew=F.smooth_l1_loss(pred_reward,rewards[:,t]/reward_scale)
            cont=F.binary_cross_entropy_with_logits(pred_continue,1-dones[:,t])
            kl=kl_divergence(posterior,prior).sum(-1).mean()
            total_rec=total_rec+rec
            total_rew=total_rew+rew
            total_cont=total_cont+cont
            total_kl=total_kl+torch.clamp(kl,min=free_nats)
        rec,rew,cont,kl=(x/steps for x in (total_rec,total_rew,total_cont,total_kl))
        loss=rec+rew+0.1*cont+kl_scale*kl
        return {"loss":loss,"recon_mse":total_pixel_mse/steps,"recon_objective":rec,"reward_smooth_l1":rew,"continue_bce":cont,"kl_nats":kl}
    @torch.no_grad()
    def predict_open_loop(self,start_image,actions):
        """Reconstruct future frames using only initial image and supplied future actions."""
        self.eval()
        state=self.observe_first(start_image)
        frames=[];rewards=[]
        for t in range(actions.shape[1]):
            state,_=self.prior_step(state,actions[:,t],sample=False)
            frame,reward,_=self.decode(state)
            frames.append(frame)
            rewards.append(reward)
        return torch.stack(frames,1),torch.stack(rewards,1)
