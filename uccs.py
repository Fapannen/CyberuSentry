import cv2
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
from typing import Literal
import torch
import argparse

from tqdm import tqdm
import pickle
import facenet_pytorch as fp

from inference import run_inference_image
from utils.bbox import crop_bbox_from_image
from utils.model import restore_model

from dist_fn.distances import CosineDistance, EuclideanDistance


def prepare_val_identities(path_to_uccs: str | Path, output_dir: str | Path) -> None:
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
        Where to store the results. The output_dir will contain 1001 
        directories with samples for each individual identity. For such
        a structure, FaceDataset class is already prepared and easily
        usable.
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
    """Build the gallery for matching input images.

    Parameters
    ----------
    path_to_gallery : str | Path
        Path to the directory containing subject identity
        directories with their images
    model : str | torch.nn.Module
        Model which is used to produce the face embeddings.
        If it is a string, a new model is instantiated, otherwise
        it is used as is
    reduction : _type_, optional
        How to merge the embeddings from a single identity.
        avg: average the embeddings
        none: return them all

    Returns
    -------
    dict[int, torch.Tensor]
        Mapping of {subject_id : reduced_embeddings}
    """
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


def gallery_lookup(
    gallery: dict[int, torch.Tensor], face_embedding: torch.Tensor, dist_fn : str, threshold=1.75
) -> int:
    """Take a given 'face_embedding' and match it against embeddings
    in a 'gallery'. Return the id of the best match in the gallery,
    if found, else -1, representing the unknown identity.

    Distance from the face embedding to each identity in the gallery
    is computed and the
    sample with minimum distance is selected as best match. If the dist of
    the best match is still higher than the threshold, it is considered an
    unknown identity and -1 is returned.

    Parameters
    ----------
    gallery : dict[int, torch.Tensor]
        A dictionary of identities. For each identity in the gallery, there
        is the identity's embedding stored, which is compared to the provided
        face embedding.
    face_embedding : torch.Tensor
        The embedding from an inferred image. This is the vector that represents
        the face from whatever image to be matched against known identities in the
        gallery.
    dist_fn
        Method of computing the distance between face embeddings. Can be either
        "cosine" or "euclidean".
    threshold : float, optional
        Determines the threshold when the identity is matched or rejected.

    Returns
    -------
    int
        ID of the identity, if matched. The ID is a key in the provided gallery.
        If no matches are found, -1 is returned.
    """
    gallery_embeddings = [gallery[subject] for subject in gallery]
    gallery_embeddings = torch.stack(gallery_embeddings)

    distance_func = EuclideanDistance() if dist_fn == "euclidean" else CosineDistance()

    print(gallery_embeddings.shape)
    print(face_embedding.shape)

    gallery_dists = torch.tensor([distance_func(subject_embedding,  face_embedding) for subject_embedding in gallery_embeddings])

    print(gallery_dists)

    best_match = torch.argmin(gallery_dists, dim=0)

    # Return the best matching ID from the gallery
    # (Argmin is done on the embeddings, but the index in UCCS gallery
    # starts with ID 1, not 0)
    return (
        list(gallery.keys())[best_match.item()]
        if gallery_dists[best_match] <= threshold  #  If matches
        else -1  # else unknown
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------
    """ Prepare dataset with cropped faces
    
    prepare_val_identities("C:/data/UCCSChallenge", "C:/data/uccs_prep")
    
    """
    # ------------------------------------------------------------------
    """ Prepare Gallery
    
    gallery_path = "C:/data/UCCSChallenge/gallery_images/gallery_images"
    model = "model-24-val-37.60191621913782"
    reduction = "avg"

    gallery = build_uccs_gallery(gallery_path, model, reduction)

    with open(f"gallery_{model.split(".")[0]}-{reduction}.pkl", "wb") as f:
        pickle.dump(gallery, f)
    
    """
    # ------------------------------------------------------------------  
    """ Search an identity from an image in UCCS gallery

    image_embedding = run_inference_image(
        "model-24-val-37.60191621913782", "img/obama.jpg", False
    )
    image_embedding = image_embedding[list(image_embedding.keys())[0]]
    gallery_file = open("gallery_model-24-val-37-avg.pkl", "rb")
    gallery = pickle.load(gallery_file)
    print(gallery_lookup(gallery, image_embedding, "euclidean"))
    """