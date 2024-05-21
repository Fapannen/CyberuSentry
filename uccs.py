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
from encoder.eva_tiny import EvaTiny
from encoder.efficientnet import EfficientNetB0

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


def gallery_similarity(
    gallery: dict[int, torch.Tensor],
    face_embedding: torch.Tensor,
    dist_fn: str,
    eucl_dist_thr: float = None,
    w: float = None,
    c: float = None,
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

    distance_func = (
        EuclideanDistance() if dist_fn == "euclidean" else torch.nn.CosineSimilarity()
    )

    gallery_dists = torch.tensor(
        [
            distance_func(subject_embedding, face_embedding).detach().item()
            for subject_embedding in gallery_embeddings
        ],
        requires_grad=False,
    )

    if dist_fn == "euclidean":
        max_dist = torch.max(gallery_dists).item()
        gallery_dists /= max_dist
        threshold = eucl_dist_thr / max_dist if eucl_dist_thr > 1.0 else eucl_dist_thr
        gallery_dists = torch.tensor(
            [
                max(1 / (1 + score), 0.75) if score <= threshold else w / (c * score)
                for score in gallery_dists.numpy()
            ],
            requires_grad=False,
        )

    return gallery_dists


def uccs_image_inference(cyberusModel, image_path) -> str:
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
        (
            i,
            faces_cropped[i],
            cyberusModel(faces_cropped[i]).detach(),
            image_faces[i],
            confidences[i],
        )
        for i in range(len(faces_cropped))
    ]

    ret_str = ""

    for _, _, gallery_distances, face_bbox, detection_score in faces_mapped:

        xmin, ymin, xmax, ymax = face_bbox
        x1 = xmin
        y1 = ymin
        width = xmax - xmin
        height = ymax - ymin

        # Format of the submission file is
        # FILE, DET_SCORE, BBX, BBY, BBW, BBH, S0001, ..., S1000
        ret_str += f"{image_path.split('/')[-1]},{detection_score},{x1},{y1},{width},{height},{','.join([str(dist) for dist in gallery_distances.detach().cpu().numpy()])}\n"

    return ret_str


def uccs_eval(model: torch.nn.Module, uccs_root: str, path_to_protocol_csv: str):
    """Run evaluation on the provided csv file (either val or test).
    This function should save the required file for the final submission.
    As such, the "model" parameter is now a fully instantiated nn.Module,
    which in its forward produces the similarities to UCCS gallery subjects.
    These outputs are used to produce the final file.

    Parameters
    ----------
    model : torch.nn.Module
        Instantiated identification model
    path_to_protocol_csv : str
        Path to the source csv which should include image path names and their detected bounding boxes.
    """
    df = pd.read_csv(path_to_protocol_csv, delimiter=",", header=0)
    # VAL: FILE,FACE_ID,SUBJECT_ID,FACE_X,FACE_Y,FACE_WIDTH,FACE_HEIGHT
    # TEST: FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y

    mode = "val" if "FACE_X" in df.columns else "test"
    if mode == "val":
        # Rename columns to the same style as in test, then handle both cases identically
        df = df.rename(
            columns={
                "FACE_X": "BB_X",
                "FACE_Y": "BB_Y",
                "FACE_WIDTH": "BB_WIDTH",
                "FACE_HEIGHT": "BB_HEIGHT",
            }
        )
        df["DETECTION_SCORE"] = 1.0

    partitions = (
        list(range(1, 9)) if mode == "val" else [f"{part:02d}" for part in range(9, 17)]
    )
    split = "validation" if mode == "val" else "test"

    # Now that we have potentially unified the dataframes, select only relevant columns
    # OUTPUT: FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,S_0001, ..., S1000
    df = df[["FILE", "DETECTION_SCORE", "BB_X", "BB_Y", "BB_WIDTH", "BB_HEIGHT"]]

    # Create placeholder columns for the subjects S_0001, ..., S_1000
    for subject_col in range(1000):
        df[f"S_{(subject_col + 1):04d}"] = np.nan

    # Now the evaluation loop. Iterate over images in the folder, find the respective rows in the df
    # and update the values there.
    for partition in partitions:
        partition_df = []
        partition_path = f"{uccs_root}/{split}_{partition}/{split}_images"
        images = os.listdir(partition_path)
        for image_path in tqdm(images, desc=f"{split} partition {partition}"):
            original_indices = df.index[df["FILE"] == image_path].tolist()
            img = cv2.cvtColor(
                cv2.imread(partition_path + "/" + image_path), cv2.COLOR_BGR2RGB
            )
            for index in original_indices:
                face_x, face_y, face_w, face_h = (
                    df.iloc[index]["BB_X"],
                    df.iloc[index]["BB_Y"],
                    df.iloc[index]["BB_WIDTH"],
                    df.iloc[index]["BB_HEIGHT"],
                )

                xmin, ymin, xmax, ymax = (
                    face_x,
                    face_y,
                    face_x + face_w,
                    face_y + face_h,
                )

                crop = crop_bbox_from_image(img, xmin, xmax, ymin, ymax)
                model_input = numpy_to_model_format(crop, add_batch_dim=True)
                model_preds = model(model_input)
                if len(model_preds.shape) >= 1:
                    model_preds = model_preds.squeeze()

                # print("Predicted subject ", torch.argmax(model_preds).item() + 1, "With score: ", model_preds[torch.argmax(model_preds).item()])

                nd = df.iloc[index].to_dict()
                for i in range(len(model_preds)):
                    # Sometimes the metric yields (although very close to 0) negative values, clip them to be safe
                    nd[f"S_{(i+1):04d}"] = (
                        str(model_preds[i].item())[:8]
                        if model_preds[i].item() >= 0.0
                        else 0.0
                    )

                partition_df.append(nd)

        partition_df = pd.DataFrame(partition_df)
        partition_df.to_csv(
            f"{split}_partition_{partition}_eval.csv", sep=",", header=True, index=False
        )


def merge_eval_csv(mode: Literal["val", "test"]):
    csv_partitions = (
        list(range(1, 9)) if mode == "val" else [f"{part:02d}" for part in range(1, 17)]
    )
    partition = "validation" if mode == "val" else "test"
    print("Loading the csvs ...")
    csvs = [
        pd.read_csv(f"{partition}_partition_{part}_eval.csv", sep=",", header=0)
        for part in csv_partitions
    ]
    print("csvs loaded.")

    final_df = pd.concat(csvs)
    final_df.to_csv(f"{partition}-final.csv", sep=",", index=False, header=True)


def square_df(csv_final_path: str):
    df = pd.read_csv(csv_final_path, sep=",", header=0)

    for col in tqdm(df.columns, desc="Squaring subject columns"):
        if str(col).startswith("S_"):
            df[col] = df[col].apply(lambda x: f"{(x**2):5f}")

    df.to_csv(
        csv_final_path.replace(".csv", "-squared.csv"),
        sep=",",
        index=False,
        header=True,
    )


if __name__ == "__main__":
    merge_eval_csv("test")
    square_df("test-final.csv")
    exit(1)
    # ------------------------------------------------------------------
    """ Prepare dataset with cropped faces
    
    prepare_val_identities("C:/data/UCCSChallenge", "C:/data/uccs_prep")
    
    """
    # ------------------------------------------------------------------
    """
    #Prepare Gallery
    
    gallery_path = "C:/data/UCCSChallenge/gallery_images/gallery_images"
    model = "model-24-val-161.36379772424698"
    reduction = "avg"

    gallery = build_uccs_gallery(gallery_path, model, reduction)

    with open(f"gallery_{model.split(".")[0]}-{reduction}.pkl", "wb") as f:
        pickle.dump(gallery, f)
    
    """
    # ------------------------------------------------------------------
    """
    #Search an identity from an image in UCCS gallery

    image_embedding = run_inference_image(
        "model-6-val-54.003331661224365", "0947_9.png", False
    )
    image_embedding = image_embedding[list(image_embedding.keys())[0]]
    gallery_file = open("gallery_model-6-val-54-avg.pkl", "rb")
    gallery = pickle.load(gallery_file)
    print(torch.argmax(gallery_similarity(gallery, image_embedding, "cosine"))) # Will not work, need to provide the model now
    """
    # ------------------------------------------------------------------
    """
    # UCCS image inference

    model = restore_model("model-30-train-6585.611407843418")
    image_path = "img/obama.jpg"
    gallery_file = open("gallery_model-24-val-37-avg.pkl", "rb")
    gallery = pickle.load(gallery_file)
    print(uccs_image_inference(gallery, model, image_path, "cosine"))
    """
    # ------------------------------------------------------------------
    # CyberuSentry
    from cyberusentry import CyberuSentry

    kerberos = CyberuSentry(
        "model-24-val-37.60191621913782",
        EvaTiny(activate=False),
        "gallery_model-24-val-37-avg.pkl",
        "model-32-val-128.81810501217842",
        EvaTiny(),
        "gallery_model-32-val-128-avg.pkl",
        "model-24-val-161.36379772424698",
        EvaTiny(),
        "gallery_model-24-val-161-avg.pkl",
    )

    uccs_eval(
        kerberos,
        "C:/data/UCCSChallenge",
        "C:/data/UCCSChallenge/protocols/protocols/UCCS-detection-baseline-test.txt",
    )
