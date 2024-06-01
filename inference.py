import facenet_pytorch as fp
import torch
import argparse
import cv2
import numpy as np

from pathlib import Path

from utils.image import read_image, numpy_to_model_format, model_format_to_numpy
from utils.model import restore_model
from utils.bbox import crop_bbox_from_image

from dist_fn.distances import EuclideanDistance, CosineDistance


def run_inference_video(
    model_path : str, video_path : str, dist_fn :str, each_nth_frame : int = 30, unique_dist_threshold : float = 0.5,
    config_name: str = "config-default"
) -> None:
    """This function analyzes the provided video file 
    using a provided model. This function creates a
    directory where all detected faces will be assigned
    to an identity. Identities are built on-the-go.
    New identity is determined based on 
    'unique_dist_threshold' where all faces that have
    its embedding distance greater than 'unique_dist_threshold'
    from all already established identities are assigned
    to a new identity. As a user, you can control how sensitive
    this boundary is. To speed things up and not infer the
    whole video file, only 'each_nth_frame' is analyzed.

    Parameters
    ----------
    model_path : str
        Path to a model checkpoint
    video_path : str
        Path to the video to be analyzed
    dist_fn : str
        "euclidean" or "cosine"
    each_nth_frame : int, optional
        Analyze each n-th frame in the video, by default 30,
        which should mean that each video is analyzed once
        per second as most videos are usually taken either
        in 24 or 30 FPS
    unique_dist_threshold : float, optional
        Threshold which determines a matching or non-matching
        identity. By default 0.5, but should be handled manually,
        cosine-based models will usually have this smaller.
    config_name
        Name of the config to instantiate the model from.
    """
    video = cv2.VideoCapture(video_path)
    video_at = 0

    # Init the face detector
    fp_model = fp.MTCNN()

    # Init the face encoder / identificator
    model = restore_model(model_path, device="cpu", config_name=config_name).eval()

    # List of unique faces
    # {int: [face1, face2]}
    uniques = {}

    # Create an output directory where identities will be shown
    video_output_dir_path = Path(f"video_inference_output/{Path(video_path).stem}")
    video_output_dir_path.mkdir(exist_ok=True, parents=True)

    distance_func = EuclideanDistance() if dist_fn == "euclidean" else CosineDistance()

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            # inference image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image_faces = fp_model.detect(image)

            if image_faces[0] is not None:
                faces_cropped = get_cropped_faces(image_faces, image)
                faces_cropped = list(
                    map(
                        lambda x: numpy_to_model_format(x, add_batch_dim=True),
                        faces_cropped,
                    )
                )
                faces_embeds = [
                    (faces_cropped[i], model(faces_cropped[i]))
                    for i in range(len(faces_cropped))
                ]

                for face_orig, face_embed in faces_embeds:
                    # Flag whether we found a match in already existing
                    found_matching = False

                    for unique in uniques:
                        if (
                            distance_func(face_embed, uniques[unique][0]).item()
                            <= unique_dist_threshold
                        ):

                            cv2.imwrite(
                                str(
                                    video_output_dir_path
                                    / str(unique)
                                    / (str(len(uniques[unique]) + 1) + ".jpg")
                                ),
                                cv2.cvtColor(
                                    model_format_to_numpy(face_orig), cv2.COLOR_RGB2BGR
                                ),
                            )

                            uniques[unique].append([face_embed])
                            found_matching = True
                            print("Found matching")
                            break

                    if not found_matching:
                        id = len(uniques)
                        print(f"Establishing a new id {id}")
                        uniques[id] = [face_embed]

                        new_id_dir = video_output_dir_path / str(id)
                        new_id_dir.mkdir(exist_ok=True, parents=True)
                        cv2.imwrite(
                            str(new_id_dir / "1.jpg"),
                            cv2.cvtColor(
                                model_format_to_numpy(face_orig), cv2.COLOR_RGB2BGR
                            ),
                        )

            video_at += each_nth_frame
            video.set(cv2.CAP_PROP_POS_FRAMES, video_at)
        else:
            video.release()
            break


def get_cropped_faces(
    detected_faces: tuple[list, list], image: np.ndarray
) -> list[np.ndarray]:
    """Create a list of faces that were cropped from the image

    Output from fp model is a tuple (bbox, confidence)
    ie. ([[tl_x, tl_y, br_x, br_y], [F2_tl_x, ...]], [[0.99], [0.45]])
    Dont forget that image is in shape (height, width)

    Parameters
    ----------
    detected_faces : tuple [bboxes, confidences]
        Output from a MTCNN model or other detector.
        If other detector is used, make sure it has
        the same structure.
    image : np.ndarray
        The original image from which to cut faces

    Returns
    -------
    List of images : list[np.ndarray]
        List of images representing the cut faces.
        The images are preprocessed to be properly
        fed into the encoder network.
    """
    cropped_faces = []
    for face_bbox in detected_faces[0]:
        xmin, ymin, xmax, ymax = face_bbox

        crop = crop_bbox_from_image(image, xmin, xmax, ymin, ymax)

        cropped_faces.append(crop)

    return cropped_faces


def run_inference_image(
    model: str | torch.nn.Module,
    image_path: str,
    save_face: bool = True,
    face_detector: torch.nn.Module | None = None,
    config_name: str = "config-default"
) -> dict[str, np.ndarray] | None:
    """Function for a complete embedding extraction from an image.
    Includes loading the model, loading the image, detecting faces,
    cropping faces, embedding the faces and returning the produced
    embeddings. The ouput is in a form of a dictionary of structure
    {image_path_face_idx_X: embedding}

    Parameters
    ----------
    model : str | torch.nn.Module
        Model which is used to produce embeddings. Can be either
        a string or an already instantiated torch Module. If it
        is string, model is loaded from the provided checkpoint,
        if it is an already instantiated torch Module, it is
        used as is.
    image_path : str
        Path to the image to be inferred
    save_face : bool, optional
        Whether to save the cropped and inferred face into
        'inference_output' directory. Useful when running manual
        inference, but we dont want to save anything during ie.
        gallery building or evaluation.
    face_detector : torch.nn.Module | None, optional
        Model which is used to detect the faces in the image.
        Similarily to 'model', if it is an already instantiated
        Module, it is used as is, otherwise a default MTCNN
        detector is used.
    config_name
        Name of the config to instantiate the model from

    Returns
    -------
    dict[str, np.ndarray] | None
        Mapping of {image_path_face_idx_X : face_embedding}.
        Each key consists of the image name and the index of the
        face within that specific image, so it is enough to
        distinguish different faces from different images and
        also within just a single image.
    """
    image_name = Path(image_path).stem

    image: np.ndarray = read_image(image_path, scale=False)

    # If an instantiated model is passed, use that one.
    # Otherwise use a default MTCNN detector
    face_detector = fp.MTCNN() if face_detector is None else face_detector

    # If an instantiated model is passed, use that one.
    # Otherwise load a checkpoint
    model = restore_model(model, device="cpu", config_name=config_name) if isinstance(model, str) else model
    model.eval()

    # Yields a tuple (bboxes, confidences)
    # ie. ([[118.15, 93.5875, 498.784, 110.2]], [[98.587]])
    # for a single detected face.
    # bbox is a quadruple of floats representing the top-left
    # and bottom-right coordinates of the bbox in the original
    # image with no scaling
    image_faces = face_detector.detect(image)
    if image_faces[0] is None:
        print(f"{image_path} has no faces detected ...")
        return None

    faces_cropped = get_cropped_faces(image_faces, image)
    faces_cropped = list(
        map(lambda x: numpy_to_model_format(x, add_batch_dim=True), faces_cropped)
    )

    # Pass the cropped faces through the identification model
    faces_mapped = [
        (i, faces_cropped[i], model(faces_cropped[i]).detach())
        for i in range(len(faces_cropped))
    ]

    # During inference on multiple images, we want to save the faces
    # to manually inspect the results.
    if save_face:
        for face_i, face_cropped, _ in faces_mapped:
            face_image_numpy = model_format_to_numpy(face_cropped)

            Path("inference_output").mkdir(exist_ok=True, parents=True)
            cv2.imwrite(
                f"inference_output/{image_name}_{face_i}.jpg",
                cv2.cvtColor(face_image_numpy, cv2.COLOR_RGB2BGR),
            )

    return {
        f"{image_name}-{face_idx}": face_embedding
        for face_idx, _, face_embedding in faces_mapped
    }


def run_inference_images(model_path: str, img1: str, img2: str, dist_fn: str, config_name: str = "config-default") -> None:
    """Run inference on two images and print out differences
    between individual faces in both images.

    Parameters
    ----------
    model_path : str
        Full path to the model which shall be used as face encoder
    img1 : str
        Path to the first image
    img2 : str
        Path to the second image
    dist_fn: str
        "euclidean" or "cosine"
    config_name
        Name of the config to instantiate the model from
    """

    distance_func = EuclideanDistance() if dist_fn == "euclidean" else CosineDistance()

    model = restore_model(model_path, device="cpu", config_name=config_name)

    img1_faces_dict = run_inference_image(model, img1, save_face=True)
    img2_faces_dict = run_inference_image(model, img2, save_face=True)

    # Store the diffs to sort them and display by similarity
    diffs = []

    all_faces = img2_faces_dict | img1_faces_dict
    all_faces_ids = list(all_faces.keys())
    print(all_faces_ids)

    for face1_idx in range(len(all_faces_ids)):
        face1_embed = all_faces[all_faces_ids[face1_idx]]
        other_faces_ids = [
            all_faces_ids[j] for j in range(face1_idx + 1, len(all_faces_ids))
        ]

        for face2_idx in range(len(other_faces_ids)):
            face2_embed = all_faces[other_faces_ids[face2_idx]]

            face_diff = distance_func(face1_embed, face2_embed)
            diffs.append(
                (all_faces_ids[face1_idx], other_faces_ids[face2_idx], face_diff)
            )

    diffs_sorted = list(sorted(diffs, key=lambda x: x[2]))

    for f1, f2, diff in diffs_sorted:
        print(f"{distance_func.__class__.__name__} of {f1} and {f2} = {diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "-m", "--model-path", action="store", dest="model_path", default="best.ckpt"
    )
    parser.add_argument("-i1", "--img1", action="store", dest="img1")
    parser.add_argument("-i2", "--img2", action="store", dest="img2")
    parser.add_argument(
        "-v",
        "--video-path",
        action="store",
        dest="video_path",
    )
    parser.add_argument(
        "-t", "--threshold", action="store", dest="threshold", default=1.0, type=float
    )
    parser.add_argument(
        "-d", "--distance", action="store", dest="dist_fn", required=True, type=str
    )
    parser.add_argument(
        "-f",
        "--frames-to-skip",
        action="store",
        dest="frames_to_skip",
        default=30,
        type=int,
    )
    parser.add_argument(
        "-c",
        "--cfg",
        action="store",
        dest="config_name",
        default="config-default",
        type=str,
    )
    args = parser.parse_args()

    if args.video_path is None and args.img1 is None:
        raise ValueError("Missing arguments ...")

    elif args.video_path is not None:
        run_inference_video(
            args.model_path,
            args.video_path,
            args.dist_fn,
            each_nth_frame=args.frames_to_skip,
            unique_dist_threshold=args.threshold,
            config_name=args.config_name
        )

    elif args.img1 is not None and args.img2 is not None:
        run_inference_images(args.model_path, args.img1, args.img2, args.dist_fn, args.config_name)

    else:
        print("Unrecognized combination of arguments, exitting ...")
