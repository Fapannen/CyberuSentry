import torch
import hydra

def restore_model(path_to_model : str, device : str = "cuda") -> torch.nn.Module:
    hydra.initialize(version_base=None, config_path="../config")
    config = hydra.compose(config_name="config-default")

    model = hydra.utils.instantiate(config.encoder.definition)
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    model.eval()

    return model
