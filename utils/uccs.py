import cv2
import torch
import numpy as np
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm


def prepare_val_identities(path_to_uccs: str | Path, output_dir: str | Path):
    """Generate a folder of cropped faces per identity to conform to
    FaceDataset class.

    The output folder contains 1001 directories - 1000 known identities and
    one -1 identity for "unknown" samples. This is easily usable by the
    FaceDataset class later on and can be seamlessly integrated into training.

    Parameters
    ----------
    path_to_uccs : str | Path
        Path to the UCCS Dataset folder. It is expected to contain extracted
        validation_<0-9> folders, containing the validation images, and 
        'protocols' folder, containing the metadata csv file with bbox
        annotations. 
    output_dir : str | Path
        Where to store the results
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    uccs_root = str(path_to_uccs)
    val_csv_path = f"{uccs_root}/protocols/protocols/validation.csv"

    df = pd.read_csv(val_csv_path, delimiter=",", header=0)

    val_partitions = list(range(1, 9))

    for partition in val_partitions:
        images = os.listdir(f"{uccs_root}/validation_{partition}/validation_images")
        for image_path in tqdm(images):

            img_df = df.loc[df["FILE"] == image_path]

            img = cv2.imread(
                f"{uccs_root}/validation_{partition}/validation_images/{image_path}"
            )
            for idx, row in img_df.iterrows():
                identity = row["SUBJECT_ID"]
                face_x, face_y, face_w, face_h = (
                    row["FACE_X"],
                    row["FACE_Y"],
                    row["FACE_WIDTH"],
                    row["FACE_HEIGHT"],
                )

                xmin, ymin, xmax, ymax = (
                    face_x,
                    face_y,
                    face_x + face_w,
                    face_y + face_h,
                )

                # Duplicate code from inference.py, #TODO clean before release
                bb_xdiff = xmax - xmin
                bb_ydiff = ymax - ymin

                new_bb_xmin = int(xmin)
                new_bb_xmin = new_bb_xmin if new_bb_xmin >= 0 else 0

                new_bb_ymin = int(ymin)
                new_bb_ymin = new_bb_ymin if new_bb_ymin >= 0 else 0

                new_bb_xmax = int(xmax)
                new_bb_xmax = (
                    new_bb_xmax if new_bb_xmax < img.shape[1] else img.shape[1] - 1
                )

                new_bb_ymax = int(ymax)
                new_bb_ymax = (
                    new_bb_ymax if new_bb_ymax < img.shape[1] else img.shape[0] - 1
                )

                crop = img[new_bb_ymin:new_bb_ymax, new_bb_xmin:new_bb_xmax, :]

                Path(str(output_dir) + "/" + str(identity)).mkdir(
                    exist_ok=True, parents=True
                )
                out_path = (
                    output_dir
                    + "/"
                    + str(identity)
                    + "/"
                    + str(row["FACE_ID"])
                    + ".jpg"
                )
                cv2.imwrite(out_path, crop)


if __name__ == "__main__":
    #prepare_val_identities("C:/data/UCCSChallenge", "C:/data/uccs_prep")