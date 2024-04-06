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

    # Transformations and augmentations
    augs = hydra.utils.instantiate(cfg.augmentations.definition)
    transforms = hydra.utils.instantiate(cfg.transforms.definition)

    # Dataset-specific definitions
    dataset = hydra.utils.instantiate(cfg.datasets.definition[0], augs=augs, transforms=transforms) # TODO: remove 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    epochs_without_improvement = 0
    best_loss = torch.inf

    for epoch in range(cfg.epochs):
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            pos1, pos2, neg = batch
            
            pos1 = pos1.to(device)
            pos2 = pos2.to(device)
            neg = neg.to(device)

            pos1_emb = encoder(pos1)
            pos2_emb = encoder(pos2)
            neg_emb = encoder(neg)

            l = loss(pos1_emb, pos2_emb, neg_emb)
            l.backward()

            optimizer.step()

            epoch_loss += l.detach().cpu().item()
        
        print(f"Epoch {epoch} loss: {epoch_loss}")

        if epoch_loss < best_loss:
            print(f"Loss improved from {best_loss} to {epoch_loss}")
            best_loss = epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.epochs_without_improvement:
                print(f"No improvement for {cfg.epochs_without_improvement} epochs. Terminating.")
                torch.save(encoder.state_dict(), cfg.best_model_path)
                return
        
        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), f"model-{epoch}-{epoch_loss}.")
            

if __name__ == "__main__":
    main()