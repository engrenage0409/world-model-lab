"""Instantiate the correct RSSM variant from a training checkpoint."""
from world_model.rssm import RSSMWorldModel
from world_model.categorical_rssm import CategoricalRSSM


def model_from_checkpoint(checkpoint):
    kind = checkpoint.get("model_type", "gaussian")
    if kind == "gaussian":
        model = RSSMWorldModel(**checkpoint["config"])
    elif kind == "categorical":
        model = CategoricalRSSM(**checkpoint["config"])
    else:
        raise ValueError(f"Unknown model_type: {kind}")
    model.load_state_dict(checkpoint["model"])
    return model
