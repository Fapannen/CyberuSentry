import facenet_pytorch as fp
import torch
import argparse
import cv2
import numpy as np

from pathlib import Path

from utils.image import read_image, numpy_to_model_format, model_format_to_numpy
from utils.model import restore_model
from utils.bbox import crop_bbox_from_image

from utils.triplet import dist


def run_inference_video(
    model_path, video_path, each_nth_frame=30, unique_dist_threshold=0.5
):
    video = cv2.VideoCapture(video_path)
    video_at = 0

    # Init the face detector
    fp_model = fp.MTCNN()

    # Init the face encoder
    model = restore_model(model_path, device="cpu").eval()

    # List of unique faces
    # {int: [face1, face2]}
    uniques = {}

    # Create an output directory where identities will be shown
    video_output_dir_path = Path(f"video_inference_output/{Path(video_path).stem}")
    # .absolute() Issues on windows :)))
    video_output_dir_path.mkdir(exist_ok=True, parents=True)

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            # inference image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imwrite("frame.jpg", frame)

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
                            dist(face_embed, uniques[unique][0])
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
        _description_
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
) -> dict[str, np.ndarray] | None:
    """Function for a complete embedding extraction from an image.
    Includes loading the model, loading the image, detecting faces,
    cropping faces, embedding the faces and returning the produced
    embeddings. The ouput is in a form of a dictionary of structure
    {image_path_face_idx: embedding}

    Parameters
    ----------
    model : str | torch.nn.Module
        _description_
    image_path : str
        _description_
    save_face : bool, optional
        _description_, by default True
    face_detector : torch.nn.Module | None, optional
        _description_, by default None

    Returns
    -------
    dict[str, np.ndarray] | None
        _description_
    """

    image_name = Path(image_path).stem

    image: np.ndarray = read_image(image_path, convert_to_tensor=False, scale=False)

    # If an instantiated model is passed, use that one.
    # Otherwise use a default MTCNN detector
    face_detector = fp.MTCNN() if face_detector is None else face_detector

    # If an instantiated model is passed, use that one.
    # Otherwise load a checkpoint
    model = restore_model(model, device="cpu") if isinstance(model, str) else model

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

    faces_mapped = [
        (i, faces_cropped[i], model(faces_cropped[i]))
        for i in range(len(faces_cropped))
    ]

    # During inference on multiple images, we want to save the faces
    # to manually inspect the results, however, when ie building
    # gallery, we dont need the image saved.
    if save_face:
        for face_i, face_cropped, face_embedding in faces_mapped:
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


def run_inference_images(model_path: str, img1: str, img2: str):
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
    """

    model = restore_model(model_path, device="cpu")

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

            face_diff = torch.sum(torch.abs(face1_embed - face2_embed))
            diffs.append(
                (all_faces_ids[face1_idx], other_faces_ids[face2_idx], face_diff)
            )

    diffs_sorted = list(sorted(diffs, key=lambda x: x[2]))

    for f1, f2, diff in diffs_sorted:
        print(f"Difference of {f1} and {f2} = {diff}")


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
        "-f",
        "--frames-to-skip",
        action="store",
        dest="frames_to_skip",
        default=30,
        type=int,
    )
    args = parser.parse_args()

    if args.video_path is None and args.img1 is None:
        raise ValueError("Missing arguments ...")

    elif args.video_path is not None:
        run_inference_video(
            args.model_path,
            args.video_path,
            each_nth_frame=args.frames_to_skip,
            unique_dist_threshold=args.threshold,
        )

    elif args.img1 is not None and args.img2 is not None:
        run_inference_images(args.model_path, args.img1, args.img2)

    else:
        print("Unrecognized combination of arguments, exitting ...")
