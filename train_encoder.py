import torch
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
import argparse

from torch.utils.data import ConcatDataset

from sampler.sampler import IdentitySampler
from utils.model import restore_model

from utils.triplet import build_triplets


@hydra.main(config_path="config", config_name="config-default", version_base=None)
def main(cfg: DictConfig):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} ...")

    # Encoder-specific definitions
    encoder = (
        hydra.utils.instantiate(cfg.encoder.definition).to(device)
        if cfg.restore_model is None
        else restore_model(cfg.restore_model, "cuda:0").to(device)
    )
    loss_fn = hydra.utils.instantiate(cfg.loss.definition)
    optimizer = hydra.utils.instantiate(
        cfg.optimizer.definition, params=encoder.parameters()
    )

    # Transformations and augmentations
    augs = hydra.utils.instantiate(cfg.augmentations.definition)
    transforms = hydra.utils.instantiate(cfg.transforms.definition)

    # Dataset-specific definitions
    train_datasets = [
        hydra.utils.instantiate(
            dataset, augs=augs, transforms=transforms, split="train"
        )
        for dataset in cfg.datasets.definition
    ]
    val_datasets = [
        hydra.utils.instantiate(dataset, augs=augs, transforms=transforms, split="val")
        for dataset in cfg.datasets.definition
    ]

    train_datasets_len = sum([len(d.get_identities()) for d in train_datasets])
    val_datasets_len = sum([len(d.get_identities()) for d in val_datasets])

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=IdentitySampler(
            train_datasets_len,
            cfg.batch_size,
            cfg.min_samples_per_id,
        ),
    )  # TODO: Move sampler config to hydra
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=IdentitySampler(
            val_datasets_len,
            cfg.batch_size,
            cfg.min_samples_per_id,
        ),
    )

    validations_without_improvement = 0
    best_val_loss = torch.inf

    start = 0 if cfg.restore_model is None else int(cfg.restore_model.split("-")[1]) + 1

    for epoch in range(start, cfg.epochs):
        if epoch == cfg.epoch_swap_to_hard:
            print(
                "Reached epoch where swap to hard strategy occurs. Resetting val_loss to account for it."
            )
            best_val_loss = torch.inf
        encoder.train()
        epoch_loss = 0
        for i, batch in enumerate(
            tqdm(train_dataloader, desc=f"epoch {epoch} train loop")
        ):
            optimizer.zero_grad()

            pos, neg = batch

            pos = pos.to(device)
            neg = neg.to(device)

            pos_emb = encoder(pos)
            neg_emb = encoder(neg)

            # Start with semi-hard samples to learn "easy"
            # ordering, and after some epochs swap to
            # pure hard samples
            if epoch < cfg.epoch_swap_to_hard:
                triplet_setting = "semi-hard"
            else:
                triplet_setting = "hard"

            triplets = build_triplets(
                pos_emb,
                neg_emb,
                cfg.min_samples_per_id,
                triplet_setting=triplet_setting,
                margin=cfg.loss.definition.margin,
            )

            loss = loss_fn(triplets[0], triplets[1], triplets[2])
            print(f"loss: {loss.detach().cpu().item()}")
            if i % 100 == 0:
                print(pos_emb[0])
            loss.backward()

            optimizer.step()

            epoch_loss += loss.detach().cpu().item()

        print(f"Epoch {epoch} train loss: {epoch_loss}")

        # For now save everything, I'm curious after each epoch
        torch.save(encoder.state_dict(), f"model-{epoch}-{epoch_loss}.")

        # Run validation
        if epoch % cfg.validation_interval == 0:
            val_loss = 0
            encoder.eval()
            for _, batch in enumerate(
                tqdm(val_dataloader, desc=f"epoch {epoch} val loop")
            ):
                pos, neg = batch

                pos = pos.to(device)
                neg = neg.to(device)

                pos_emb = encoder(pos)
                neg_emb = encoder(neg)

                # As recommended, evaluate on hard only
                # to see the improvements
                triplets = build_triplets(
                    pos_emb,
                    neg_emb,
                    cfg.min_samples_per_id,
                    triplet_setting="hard",
                    margin=cfg.loss.definition.margin,
                )

                loss = loss_fn(triplets[0], triplets[1], triplets[2])

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


if __name__ == "__main__":
    main()
