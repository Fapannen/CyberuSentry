import cv2
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
from typing import Literal
import torch

from tqdm import tqdm
import pickle
import facenet_pytorch as fp

from inference import run_inference_image
from utils.bbox import crop_bbox_from_image
from utils.model import restore_model


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

            # No need to use 'read_image' as nothing is being detected
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

                crop = crop_bbox_from_image(img, xmin, xmax, ymin, ymax)

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


def build_uccs_gallery(
    path_to_gallery: str | Path,
    model: str | torch.nn.Module,
    reduction=Literal["none", "avg"],
) -> dict[int, torch.Tensor]:
    gallery = {}

    model = restore_model(model, device="cpu") if isinstance(model, str) else model
    face_detector = fp.MTCNN()

    for subject in tqdm(
        os.listdir(path_to_gallery),
        desc="Building gallery",
        total=len(os.listdir(path_to_gallery)),
    ):
        subject_id = int(subject)
        subject_embeddings = []

        for subject_image in os.listdir(f"{path_to_gallery}/{subject}"):
            subject_embedding: dict[str, torch.Tensor] = run_inference_image(
                model,
                f"{path_to_gallery}/{subject}/{subject_image}",
                save_face=False,
                face_detector=face_detector,
            )

            # Inference can return None if no face has been detected
            if subject_embedding is None:
                continue

            # Here we really expect only a single face per image.
            if len(subject_embedding) != 1:
                print(f"Wrong number of detections in subject {subject_id}")
                continue

            detected_faces = subject_embedding[list(subject_embedding.keys())[0]]

            subject_embeddings.append(detected_faces[0])

        subject_embeddings = torch.stack(subject_embeddings)

        match reduction:
            case "avg":
                gallery[subject_id] = torch.mean(
                    subject_embeddings, dim=0, keepdim=True
                )
            case _:
                gallery[subject_id] = subject_embeddings

    return gallery


if __name__ == "__main__":
    # prepare_val_identities("C:/data/UCCSChallenge", "C:/data/uccs_prep")

    gallery_path = "C:/data/UCCSChallenge/gallery_images/gallery_images"
    model = "model-24-val-37.60191621913782"
    reduction = "avg"

    gallery = build_uccs_gallery(gallery_path, model, reduction)

    with open(f"gallery_{model.split(".")[0]}-{reduction}.pkl", "wb") as f:
        pickle.dump(gallery, f)
