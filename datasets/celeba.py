import torchvision

from typing import Literal

from datasets.face_dataset import FaceDataset


class CelebADataset(FaceDataset):
    """CelebA dataset
    https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    """

    def __init__(
        self,
        path_to_dataset_dir: str,
        augs: torchvision.transforms.v2.Compose | None,
        transforms: torchvision.transforms.v2.Compose,
        split: Literal["train", "val"],
        train_split: float = 0.8,
        path_to_identity_annotations: str = None,
    ):
        self.path_to_identity_annotations = path_to_identity_annotations
        super().__init__(path_to_dataset_dir, augs, transforms, split, train_split)

    def build_identities(self) -> dict[int, list[str]]:
        # Dataset consists of 202 599 images across 10 177 identities.
        # Images do not have any naming and go from 00000.jpg to max.jpg
        # The identity annotations are provided in a separate file of form
        # <image_name> <identity_id>
        # Build a dictionary {id: [name_img_path1, name_img_path2, ...]}

        identities: dict[int, list[str]] = {}
        with open(self.path_to_identity_annotations, "r") as annotations_file:
            for line in annotations_file.readlines():
                image_path, identity = line.replace("\n", "").split(" ")
                identity = int(identity)

                if identity not in identities:
                    identities[identity] = [self.dataset_dir + image_path]
                else:
                    identities[identity].append(self.dataset_dir + image_path)
        return identities
