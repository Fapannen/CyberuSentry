import torch
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config-default", version_base=None)
def main(cfg : DictConfig):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} ...")

    # Encoder-specific definitions
    encoder = hydra.utils.instantiate(cfg.encoder.definition).to(device)
    loss = hydra.utils.instantiate(cfg.loss.definition)
    optimizer = hydra.utils.instantiate(cfg.optimizer.definition, params = encoder.parameters())

    # Dataset-specific definitions
    dataset = hydra.utils.instantiate(cfg.datasets.definition[0]) # TODO: remove 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Transformations and augmentations
    augs = hydra.utils.instantiate(cfg.augmentations.definition)
    transforms = hydra.utils.instantiate(cfg.transforms.definition)

    for epoch in range(cfg.epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            pos1, pos2, neg = batch
            
            pos1 = augs(transforms(pos1.to(device)))
            pos2 = augs(transforms(pos2.to(device)))
            neg = augs(transforms(neg.to(device)))

            pos1_emb = encoder(pos1)
            pos2_emb = encoder(pos2)
            neg_emb = encoder(neg)

            l = loss(pos1_emb, pos2_emb, neg_emb)
            l.backward()

            optimizer.step()

            print(f"loss: {l.detach().cpu().item()}")

if __name__ == "__main__":
    main()