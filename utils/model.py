import torch
import hydra


def restore_model(
    path_to_model: str, device: str = "cuda", config_name: str = "config-default"
) -> torch.nn.Module:
    """Load a model from a checkpoint and return the model as a runnable
    instance.

    Parameters
    ----------
    path_to_model : str
        Full path to the model checkpoint
    device : str, optional
        On which torch.device to load Defaults to "cuda".
    config_name : str, optional
        Name of the config which was used to construct
         the loaded model. This is necessary to know
         what model class to instantiate for successful
         loading of the weights.
         Defaults to "config-default"

    Returns
    -------
    torch.nn.Module
        Restored model for inference. Note that no '.eval()'
        is called here. Remember to do so when using this
        for inference.
    """
    try:
        hydra.initialize(version_base=None, config_path="../config")
    except:
        print(
            """Failed to initialize hydra. It might be already
               initialized which causes an error. Attempting without
               initializing ..."""
        )
    finally:
        config = hydra.compose(config_name=config_name)

    model = hydra.utils.instantiate(config.encoder.definition)
    model.load_state_dict(torch.load(path_to_model, map_location=device))

    return model
