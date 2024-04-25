import os

from pathlib import Path

from datasets.face_dataset import FaceDataset


class UCCS(FaceDataset):
    """UCCSChallenge dataset
    https://vast.uccs.edu/Opensetface/
    """

    def build_identities(self) -> dict[int, list[str]]:
        identities: dict[int, list[str]] = {}

        for identity_subdir in os.listdir(self.dataset_dir):
            # subdirs are named [-1, 1000]
            identity_identifier = int(identity_subdir)

            # Label -1 is meant for unknown identities, so keep it out
            if identity_identifier == -1:
                continue

            identities[identity_identifier] = []
            for image_path in os.listdir(str(Path(self.dataset_dir) / identity_subdir)):
                identities[identity_identifier].append(
                    str(Path(self.dataset_dir) / identity_subdir / image_path)
                )

        return identities
