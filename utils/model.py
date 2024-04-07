import torch
import hydra


def restore_model(
    path_to_model: str, device: str = "cuda", config_name: str = "config-default"
) -> torch.nn.Module:
    hydra.initialize(version_base=None, config_path="../config")
    config = hydra.compose(config_name=config_name)

    model = hydra.utils.instantiate(config.encoder.definition)
    model.load_state_dict(torch.load(path_to_model, map_location=device))

    return model
