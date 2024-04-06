import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config-default", version_base=None)
def main(cfg : DictConfig):
    encoder = hydra.utils.instantiate(cfg.encoder.definition)
    dataset = hydra.utils.instantiate(cfg.datasets.definition)
    print("All good ...")
    return

if __name__ == "__main__":
    main()