"""Instantiate the correct RSSM variant from a training checkpoint."""
from world_model.rssm import RSSMWorldModel
from world_model.categorical_rssm import CategoricalRSSM
from world_model.v3_rssm import V3RSSM


def model_from_checkpoint(checkpoint):
    kind = checkpoint.get("model_type", "gaussian")
    if kind == "gaussian":
        model = RSSMWorldModel(**checkpoint["config"])
    elif kind == "categorical":
        model = CategoricalRSSM(**checkpoint["config"])
    elif kind == "v3_experimental":
        model = V3RSSM(**checkpoint["config"])
    else:
        raise ValueError(f"Unknown model_type: {kind}")
    model.load_state_dict(checkpoint["model"])
    return model
