import os

from datasets.face_dataset import FaceDataset


class LFWDataset(FaceDataset):
    """Labeled Faces in the Wild dataset from Kaggle
    https://www.kaggle.com/datasets/atulanandjha/lfwpeople/data
    """

    def build_identities(self) -> dict[int, list[str]]:
        # Each directory has images of the same person
        # Build a dictionary {name: [name_img_path1, name_img_path2, ...]}
        identities = {
            f: [
                self.dataset_dir + f + "/" + file
                for file in os.listdir(self.dataset_dir + f)
            ]
            for f in os.listdir(self.dataset_dir)
            if os.path.isdir(self.dataset_dir + f)
        }

        new_identities = {}

        for i in range(len(list(identities.keys()))):
            new_identities[i] = identities[list(identities.keys())[i]]

        return new_identities
