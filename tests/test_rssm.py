import torch
from world_model.rssm import RSSMWorldModel

def test_loss_gradients():
    torch.manual_seed(0)
    model=RSSMWorldModel(hidden=32,stoch=8,embed=32)
    images=torch.rand(1,3,3,96,96)
    actions=torch.rand(1,2,3)
    rewards=torch.zeros(1,2)
    dones=torch.zeros(1,2)
    output=model.loss(images,actions,rewards,dones)
    assert output["loss"].ndim==0
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.gru.weight_ih.grad is not None
    assert model.decoder.net[0].weight.grad is not None

def test_prior_shape():
    model=RSSMWorldModel(hidden=32,stoch=8,embed=32)
    frame=torch.rand(1,3,96,96)
    actions=torch.zeros(1,2,3)
    predicted,rewards=model.predict_open_loop(frame,actions)
    assert predicted.shape==(1,2,3,96,96)
    assert rewards.shape==(1,2)
