import os

from pathlib import Path

from datasets.face_dataset import FaceDataset


class CasiaWebface(FaceDataset):
    """Casia Webface dataset
    https://www.kaggle.com/datasets/ntl0601/casia-webface
    """

    def build_identities(self) -> dict[int, list[str]]:
        identities = {}

        for identity_subdir in os.listdir(self.dataset_dir):
            # subdirs are named ie "0000001", "0000002", ...
            identity_identifier = int(identity_subdir)
            identities[identity_identifier] = []
            for image_path in os.listdir(str(Path(self.dataset_dir) / identity_subdir)):
                identities[identity_identifier].append(
                    str(Path(self.dataset_dir) / identity_subdir / image_path)
                )

        return identities
