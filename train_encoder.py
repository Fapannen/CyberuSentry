import torch
import hydra
from tqdm import tqdm
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config-default", version_base=None)
def main(cfg: DictConfig):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} ...")

    # Encoder-specific definitions
    encoder = hydra.utils.instantiate(cfg.encoder.definition).to(device)
    loss_fn = hydra.utils.instantiate(cfg.loss.definition)
    optimizer = hydra.utils.instantiate(
        cfg.optimizer.definition, params=encoder.parameters()
    )

    # Transformations and augmentations
    augs = hydra.utils.instantiate(cfg.augmentations.definition)
    transforms = hydra.utils.instantiate(cfg.transforms.definition)

    # Dataset-specific definitions
    train_dataset = hydra.utils.instantiate(
        cfg.datasets.definition[0], augs=augs, transforms=transforms, split="train"
    )  # TODO: remove 0
    val_dataset = hydra.utils.instantiate(
        cfg.datasets.definition[0], augs=augs, transforms=transforms, split="val"
    )  # TODO: remove 0
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=True
    )

    validations_without_improvement = 0
    best_val_loss = torch.inf

    for epoch in range(cfg.epochs):
        encoder.train()
        epoch_loss = 0
        for _, batch in enumerate(
            tqdm(train_dataloader, desc=f"epoch {epoch} train loop")
        ):
            optimizer.zero_grad()

            pos1, pos2, neg = batch

            pos1 = pos1.to(device)
            pos2 = pos2.to(device)
            neg = neg.to(device)

            pos1_emb = encoder(pos1)
            pos2_emb = encoder(pos2)
            neg_emb = encoder(neg)

            loss = loss_fn(pos1_emb, pos2_emb, neg_emb)
            print(f"loss: {loss.detach().cpu().item()}")
            loss.backward()

            optimizer.step()

            epoch_loss += loss.detach().cpu().item()

        print(f"Epoch {epoch} train loss: {epoch_loss}")

        # Run validation
        if epoch % cfg.validation_interval == 0:
            val_loss = 0
            encoder.eval()
            for _, batch in enumerate(
                tqdm(val_dataloader, desc=f"epoch {epoch} val loop")
            ):
                pos1, pos2, neg = batch

                pos1 = pos1.to(device)
                pos2 = pos2.to(device)
                neg = neg.to(device)

                pos1_emb = encoder(pos1)
                pos2_emb = encoder(pos2)
                neg_emb = encoder(neg)

                loss = loss_fn(pos1_emb, pos2_emb, neg_emb)

                val_loss += loss.detach().cpu().item()

            print(f"Epoch {epoch} val loss: {val_loss}")

            if val_loss < best_val_loss:
                print(f"Val loss improved from {best_val_loss} to {val_loss}")
                best_val_loss = val_loss
                validations_without_improvement = 0

                # Save the model as best
                torch.save(encoder.state_dict(), cfg.best_model_path)
            else:
                validations_without_improvement += 1
                if (
                    validations_without_improvement
                    >= cfg.validations_without_improvement
                ):
                    print(
                        f"No improvement for {cfg.validations_without_improvement} epochs. Terminating."
                    )
                    return

        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), f"model-{epoch}-{epoch_loss}.")


if __name__ == "__main__":
    main()
