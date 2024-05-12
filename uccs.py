import cv2
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
from typing import Literal
import torch
import argparse
import numpy as np
import copy

from tqdm import tqdm
import pickle
import facenet_pytorch as fp

from inference import run_inference_image, get_cropped_faces
from utils.image import read_image, numpy_to_model_format, model_format_to_numpy
from utils.model import restore_model
from utils.bbox import crop_bbox_from_image

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


def gallery_distance(
    gallery: dict[int, torch.Tensor], face_embedding: torch.Tensor, dist_fn : str, eucl_dist_thr : float = None, w : float = None, c : float = None
) -> torch.Tensor:
    """Take a given 'face_embedding' and match it against embeddings
    in a 'gallery'. Return the computed distances of the face_embed
    compared to all subjects in the gallery.

    In case of euclidean distance as the distance function, the
    distances are first max-normalized into [0, 1] range and then
    the "matching" subjects are selected depending on a threshold.
    Subjects with distance lower than threshold are considered matching,
    subjects with distance larger than threshold are considered non-matching.

    Output from this function is a torch.Tensor containing the distances
    of the provided face embed to all face embeddings in the gallery and
    the values are in the [0, 1] range.

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
    eucl_dist_thr
        Used in the case of euclidean distance function. Determines the threshold
        under which the distances are considered matching
    w
        Controls the value at which the non-matching samples scores start
    c
        Controls the scale of the diminishing of scores in the case of non-matching
        samples

    Returns
    -------
    torch.Tensor
        Tensor of distances of the embedding to all subjects in the gallery
    """
    gallery_embeddings = [gallery[subject] for subject in gallery]
    gallery_embeddings = torch.stack(gallery_embeddings)

    distance_func = EuclideanDistance() if dist_fn == "euclidean" else CosineDistance()

    gallery_dists = torch.tensor([distance_func(subject_embedding,  face_embedding).detach().item() for subject_embedding in gallery_embeddings], requires_grad=False)
    
    if dist_fn == "euclidean":
        max_dist = torch.max(gallery_dists).item()
        gallery_dists /= max_dist
        threshold = eucl_dist_thr / max_dist if eucl_dist_thr > 1.0 else eucl_dist_thr
        gallery_dists = torch.tensor([1 / (1 + score) if score <= threshold else w / (c * score) for score in gallery_dists.numpy()],requires_grad=False)
    
    return gallery_dists

def uccs_image_inference(gallery, model, image_path, dist_fn : str, eucl_dist_thr : float = 0.2, w : float = 0.4, c : float = 4.0) -> str:
    """Perform an inference on an image and output
    a string representing rows in the submission
    file.

    # TODO: Docstrings

    Parameters
    ----------
    gallery : dict[int, torch.Tensor]
        A dictionary of identities. For each identity in the gallery, there
        is the identity's embedding stored, which is compared to the provided
        face embedding.
    model : nn.Module
        Model that should be used for the inference
    image_path : str
        Path to the image to be inferred
    dist_fn
        Which distance function to use when computing gallery similarity
    """
    image = read_image(image_path, convert_to_tensor=False, scale=False)

    # Init same settings as UCCS baseline detector
    face_detector = fp.MTCNN(thresholds=[0.2, 0.2, 0.2])

    image_faces, confidences = face_detector.detect(image)
    if image_faces is None:
        print(f"{image_path} has no faces detected ...")
        return ""

    faces_cropped = get_cropped_faces([image_faces], image)
    faces_cropped = list(
        map(lambda x: numpy_to_model_format(x, add_batch_dim=True), faces_cropped)
    )

    faces_mapped = [
        (i, faces_cropped[i], model(faces_cropped[i]).detach(), image_faces[i], confidences[i])
        for i in range(len(faces_cropped))
    ]

    ret_str = ""

    for _, _, face_embed, face_bbox, detection_score in faces_mapped:
        if dist_fn == "euclidean":
            gallery_distances = gallery_distance(gallery, face_embed, dist_fn, eucl_dist_thr=eucl_dist_thr, w=w, c=c)
        else:
            gallery_distances = gallery_distance(gallery, face_embed, dist_fn)

        xmin, ymin, xmax, ymax = face_bbox
        x1 = xmin
        y1 = ymin
        width = xmax - xmin
        height = ymax - ymin

        # Format of the submission file is
        # FILE, DET_SCORE, BBX, BBY, BBW, BBH, S0001, ..., S1000
        ret_str += f"{image_path.split('/')[-1]},{detection_score},{x1},{y1},{width},{height},{','.join([str(dist) for dist in gallery_distances.detach().cpu().numpy()])}\n"

    return ret_str





    

    




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
    # ------------------------------------------------------------------
    # UCCS image inference

    model = restore_model("model-24-val-37.60191621913782")
    image_path = "img/obama.jpg"
    gallery_file = open("gallery_model-24-val-37-avg.pkl", "rb")
    gallery = pickle.load(gallery_file)
    print(uccs_image_inference(gallery, model, image_path, "cosine"))