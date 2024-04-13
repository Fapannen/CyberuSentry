import torch
import torchvision
import os
import random

from pathlib import Path
from typing import Literal
from torch.utils.data import Dataset
from utils.image import read_image


class CelebADataset(Dataset):
    """CelebA dataset
    https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    """

    def __init__(
        self,
        path_to_dataset_dir: str,
        path_to_identity_annotations: str,
        augs: torchvision.transforms.v2.Compose | None,
        transforms: torchvision.transforms.v2.Compose,
        split: Literal["train", "val"],
        train_split: float = 0.8,
    ):
        """
        Parameters
        ----------
        path_to_dataset_dir : str
            Path to the directory containing the images
        path_to_identity_annotations : str
            Path to the identity annotations file
        augs : torchvision.transforms.v2.Compose | None
            Optional augmentations. Can be None in validation
        transforms : torchvision.transforms.v2.Compose
            Transformations to apply to images
        split : Literal["train", "val"]
            Which split is being run
        train_split : float, optional
            train/val ratio, by default 0.8
        """
        self.dataset_dir = (
            path_to_dataset_dir
            if path_to_dataset_dir.endswith("/")
            else path_to_dataset_dir + "/"
        )

        # Dataset consists of 202 599 images across 10 177 identities.
        # Images do not have any naming and go from 00000.jpg to max.jpg
        # The identity annotations are provided in a separate file of form
        # <image_name> <identity_id>
        # Build a dictionary {id: [name_img_path1, name_img_path2, ...]}

        self.identities = {}
        with open(path_to_identity_annotations, "r") as annotations_file:
            for line in annotations_file.readlines():
                image_path, identity = line.replace("\n", "").split(" ")
                identity = int(identity)

                if identity not in self.identities:
                    self.identities[identity] = [self.dataset_dir + image_path]
                else:
                    self.identities[identity].append(self.dataset_dir + image_path)

        # Build a train / val split
        self.split = split
        identities = list(self.identities.keys())
        num_identities = len(identities)

        split_identities = (
            identities[: int(train_split * num_identities)]
            if split == "train"
            else identities[int(train_split * num_identities) :]
        )

        self.identities = {
            k: v for k, v in self.identities.items() if k in split_identities
        }

        # Identities with more than one image samples
        self.mto_identities = [
            identity
            for identity in self.identities
            if len(self.identities[identity]) > 1
        ]
        self.num_mto_identities = len(self.mto_identities)

        # Augmentations and transformations
        self.augs = augs
        self.transforms = transforms

    def __len__(self):
        # Represent one epoch as one cycle per each "usable" identity
        return self.num_mto_identities

    def get_identities(self):
        # TODO: Add iterating over images per identity instead of
        # just picking sth at random
        return self.mto_identities.copy()

    def get_same_identity_tuple(self, origin_identity: str) -> tuple[str, str]:
        """Construct a tuple (img, img) of images of the same identity.

        Parameters
        ----------
        origin_identity : str
            Identifier of the identity to generate tuple from

        Returns
        -------
        tuple[str, str]
            Paths to two different images of the same identity
        """
        identity_samples = self.identities[origin_identity]
        sample1 = random.choice(identity_samples)
        sample2 = random.choice([s for s in identity_samples if s != sample1])
        return sample1, sample2

    def get_same_identity(self, identity: str) -> str:
        return random.choice(self.identities[identity])

    def get_different_identity(self, origin_identity: str) -> tuple[str, int]:
        """Get one sample of a different identity than was used to
        construct a positive tuple for training. Note that here we
        can utilize both 'mto' and 'o'

        Parameters
        ----------
        origin_identity : str
            Identity to exclude

        Returns
        -------
        str
            Path to an image of a different identity than was provided
        """
        different_identity = random.choice(
            [identity for identity in self.identities if identity != origin_identity]
        )

        return random.choice(self.identities[different_identity]), different_identity

    def __getitem__(self, idx):
        """Returns a training tuple for triplet loss

        positive images are images of the same identity, whereas
        negative image is a different identity.

        Parameters
        ----------
        idx : _type_
            Index of the item

        Returns
        -------
        _type_
            A tuple (pos_img, pos_img2, neg_image) which is designed
            to be used for triplet loss
        """
        origin_identity = self.mto_identities[idx]

        # Get tuple of the same identity
        pos_img = self.get_same_identity(origin_identity)

        # Get another identity
        neg_img, neg_identity = self.get_different_identity(origin_identity)

        pos_img = read_image(pos_img)
        neg_img = read_image(neg_img)

        if self.split == "train":
            pos_img = self.augs(pos_img)
            neg_img = self.augs(neg_img)

        pos_img = self.transforms(pos_img)
        neg_img = self.transforms(neg_img)

        # TODO: Identities are probably not needed anymore
        return pos_img, origin_identity, neg_img, neg_identity
