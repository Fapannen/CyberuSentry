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
from utils.gallery import gallery_similarity
from encoder.eva_tiny import EvaTiny
from encoder.efficientnet import EfficientNetB0
from encoder.mobilenet import MobileNetV2

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

            # No need to use 'read_image' as nothing is being detected and
            # conversion from BGR to RGB is not necessary.
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


def uccs_image_inference(cyberusModel : torch.nn.Module, image_path : str) -> str:
    """Perform an inference on an image and output
    a string representing rows in the submission
    file for the given image. The string conforms
    to the required format of the UCCS challenge.

    Parameters
    ----------
    cyberusModel : torch.nn.Module
        Instantiated CyberuSentry model with UCCS gallery
        loaded.
    image_path : str
        Path to the image to be inferred

    Returns
    ----------
    str
        Row(s) to be appended to the submission file.
        Can consist of multiple rows, if multiple
        detections were found in the image.
    """
    image = read_image(image_path, scale=False)

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


def uccs_eval(model: torch.nn.Module, uccs_root: str, path_to_protocol_csv: str) -> None:
    """Run evaluation on the provided csv file (either val or test).
    This function should save the required file for the final submission.
    As such, the "model" parameter is now a fully instantiated nn.Module,
    which in its forward produces the similarities to UCCS gallery subjects.
    
    Since the baseline csv files provided for validation and testing
    are split into multiple partitions (1-9 for val, 1-17 for test),
    this function is applied per-partition and yields csv files
    with the predictions for each partition. As such, the final
    csv for submission must be created by merging the partitions.

    Parameters
    ----------
    model : torch.nn.Module
        Instantiated identification model
    path_to_protocol_csv : str
        Path to the source csv which should include
        image path names and their detected bounding
        boxes.
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
        list(range(1, 9)) if mode == "val" else [f"{part:02d}" for part in range(1, 17)]
    )
    split = "validation" if mode == "val" else "test"

    # Now that we ensured that the dataframes are unified, select only relevant columns
    # SUBMISSION OUTPUT FORMAT: FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,S_0001, ..., S1000
    # Keep only the following columns, subject scores will be computed soon.
    df = df[["FILE", "DETECTION_SCORE", "BB_X", "BB_Y", "BB_WIDTH", "BB_HEIGHT"]]

    # Create placeholder columns for the subjects S_0001, ..., S_1000
    for subject_col in range(1000):
        df[f"S_{(subject_col + 1):04d}"] = np.nan

    # Now the evaluation loop. Iterate over images in the folder, find the respective rows in the df
    # and update the values there. Iterating over rows in the df would be much slower, since we would
    # need to open and close a given image several times. Considering the images are high resolution,
    # it would be painfully slow.
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

                # For debugging purposes, can come in handy
                #print("Predicted subject ", torch.argmax(model_preds).item() + 1, "With score: ", model_preds[torch.argmax(model_preds).item()])

                # Get the row of the detection
                nd = df.iloc[index].to_dict()

                # Flush the predictions into the row
                for i in range(len(model_preds)):  
                    nd[f"S_{(i+1):04d}"] = (
                        str(model_preds[i].item())[:8]
                    )

                partition_df.append(nd)

        partition_df = pd.DataFrame(partition_df)
        partition_df.to_csv(
            f"{split}_partition_{partition}_eval.csv", sep=",", header=True, index=False
        )


def merge_eval_csv(mode: Literal["val", "test"]) -> None:
    """Merge the csv files produced by 'uccs_eval' function.
    Since the evaluation data was provided by the organizers
    in several partitions, the partitions need to be merged
    together to produce the final submission file.

    Parameters
    ----------
    mode : Literal["val", "test"]
        Csv files of which stage to merge together
    """
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


def square_df(csv_final_path: str) -> None:
    """One of my experiments was to try to
    square the predictions of the models.
    The organizers state in the instructions
    that assigning high confidences to misdetections
    or wrong identities 'will lead to penalties'. By
    squaring the predictions, the small values will
    get shrunk and the most confident predictions will
    remain strong.

    Parameters
    ----------
    csv_final_path : str
        Path to the final csv file.
    """
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


def shift_values(csv_path : str, direction="left") -> None:
    """As pointed out by one of the organizers,
    the expected predictions in the submission
    files are Cosine Similarities, which fall in
    the [-1, 1] range. Since my models are designed
    to produce predictions in [0, 1] range, I need
    to shift the final csv files' predictions before 
    the submission. It would not be a disaster per se,
    the evaluation should still work, but let's make
    sure everything is correct :)

    Parameters
    ----------
    csv_path : str
        Path to the uccs csv file
    direction : str, optional
        left:
            Shift from [0, 1] to [-1, 1]
        right:
            Shift from [1, 1] to [0, 1]
    """
    df = pd.read_csv(csv_path, sep=",", header=0)

    for col in tqdm(df.columns, desc=f"Shifting subject columns to the {direction}"):
        if str(col).startswith("S_"):
            df[col] = df[col].apply(
                lambda x: f"{((x * 2) - 1):5f}" if direction == "left" else  f"{((x + 1) / 2):5f}"
            )

    df.to_csv(
        csv_path.replace(".csv", "-shifted.csv"),
        sep=",",
        index=False,
        header=True,
    )


if __name__ == "__main__":
    """
    This file was used in quite a chaotic and unstructured manner due to
    increased time pressure. Since I do not expect this file to be
    worked upon in the future, I'll leave it as is with examples how
    I worked with it.
    
    -------------------------------------------------------------------
    1) PREPARE DATASET WITH CROPPED FACES CONFORMING TO FACE_DATASET
       CLASS (CROP THE FACES AND SAVE THEM AS INDIVIDUAL IMAGES IN 
       THE APPROPRIATE FOLDER)
    
    prepare_val_identities("C:/data/UCCSChallenge", "C:/data/uccs_prep")
    -------------------------------------------------------------------
    2) PREPARE THE UCCS GALLERY FILE (ONCE EACH HEAD COMPLETED ITS TRAINING,
    ITS GALLERY HAD TO BE COMPUTED FOR USE IN THE FINAL ENSEMBLE)
    
    gallery_path = "C:/data/UCCSChallenge/gallery_images/gallery_images"
    model = "model-6-val-0.0"
    reduction = "avg"

    gallery = build_uccs_gallery(gallery_path, model, reduction)

    with open(f"gallery_{model.split(".")[0]}-{reduction}.pkl", "wb") as f:
        pickle.dump(gallery, f)
    ------------------------------------------------------------------
    3) SEARCH AN IDENTITY FROM AN IMAGE IN THE UCCS GALLERY (RUN THE
       FULL PROCESS. LOAD AN IMAGE -> CROP FACES -> IDENTIFY FACES -> OUTPUT
       PREDICTION)

    image_embedding = run_inference_image(
        "model-6-val-54.003331661224365", "0947_9.png", False
    )
    image_embedding = image_embedding[list(image_embedding.keys())[0]]
    gallery_file = open("gallery_model-6-val-54-avg.pkl", "rb")
    gallery = pickle.load(gallery_file)
    print(torch.argmax(gallery_similarity(gallery, image_embedding, "cosine")))
    ------------------------------------------------------------------
    4) FINAL EVALUATION PIPELINE (USED IN THE FINAL SUBMISSION)
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
        "C:/data/UCCSChallenge/protocols/protocols/validation.csv",
    )

    uccs_eval(
        kerberos,
        "C:/data/UCCSChallenge",
        "C:/data/UCCSChallenge/protocols/protocols/UCCS-detection-baseline-test.txt",
    )
    
    merge_eval_csv("val")
    merge_eval_csv("test")
    square_df("validation-final.csv")
    square_df("test-final.csv")
    
    shift_values("test-final.csv")
    shift_values("test-final-squared.csv")
    shift_values("validation-final.csv")
    shift_values("validation-final-squared.csv")
    """
