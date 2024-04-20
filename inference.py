import facenet_pytorch as fp
import torch
import argparse
import cv2
import numpy as np

from pathlib import Path

from utils.image import read_image, numpy_to_model_format, model_format_to_numpy
from utils.model import restore_model

from utils.triplet import dist


def run_inference_video(model_path, video_path, each_nth_frame = 30, unique_dist_threshold = 0.5):
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
    video_output_dir_path = Path(f"video_inference_output/{Path(video_path).stem}").absolute()
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
                faces_cropped = list(map(lambda x: numpy_to_model_format(x, add_batch_dim=True), faces_cropped))
                faces_embeds = [
                    (faces_cropped[i], model(faces_cropped[i]))
                    for i in range(len(faces_cropped))
                ]

                for face_orig, face_embed in faces_embeds:
                    # Flag whether we found a match in already existing
                    found_matching = False

                    for unique in uniques:
                        if dist(face_embed, uniques[unique][0]) <= unique_dist_threshold:

                            cv2.imwrite(
                                str(video_output_dir_path / str(unique) / (str(len(uniques[unique]) + 1) + ".jpg")),
                                cv2.cvtColor(model_format_to_numpy(face_orig), cv2.COLOR_RGB2BGR)
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
                                cv2.cvtColor(model_format_to_numpy(face_orig), cv2.COLOR_RGB2BGR)
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

        bb_xdiff = xmax - xmin
        bb_ydiff = ymax - ymin

        new_bb_xmin = int(xmin - (bb_xdiff // 8))
        new_bb_xmin = new_bb_xmin if new_bb_xmin >= 0 else 0

        new_bb_ymin = int(ymin - (bb_ydiff // 8))
        new_bb_ymin = new_bb_ymin if new_bb_ymin >= 0 else 0

        new_bb_xmax = int(xmax + (bb_xdiff // 8))
        new_bb_xmax = (
            new_bb_xmax if new_bb_xmax < image.shape[1] else image.shape[1] - 1
        )

        new_bb_ymax = int(ymax + (bb_ydiff // 8))
        new_bb_ymax = (
            new_bb_ymax if new_bb_ymax < image.shape[1] else image.shape[0] - 1
        )

        crop = image[new_bb_ymin:new_bb_ymax, new_bb_xmin:new_bb_xmax, :]

        cropped_faces.append(crop)

    return cropped_faces


def run_inference_images(model_path: str, img1: str, img2: str):
    """Run inference on two images

    TODO: Decide whether the function should output the diffs or
    the number of unique faces ...

    Parameters
    ----------
    model_path : str
        Full path to the model which shall be used as face encoder
    img1 : str
        Path to the first image
    img2 : str
        Path to the second image
    """
    img1: np.ndarray = read_image(img1, convert_to_tensor=False, scale=False)
    img2: np.ndarray = read_image(img2, convert_to_tensor=False, scale=False)

    fp_model = fp.MTCNN()

    # Yields a tuple (bboxes, confidences)
    # ie. ([[118.15, 93.5875, 498.784, 110.2]], [[98.587]])
    # for a single detected face.
    # bbox is a quadruple of floats representing the top-left
    # and bottom-right coordinates of the bbox in the original
    # image with no scaling
    img1_faces = fp_model.detect(img1)
    img2_faces = fp_model.detect(img2)

    if img1_faces[0] is None or img2_faces[0] is None:
        print("No faces detected in one of the images!")
        print("Exitting ...")
        return

    img1_faces_cropped = get_cropped_faces(img1_faces, img1)
    img2_faces_cropped = get_cropped_faces(img2_faces, img2)

    faces_cropped = img1_faces_cropped + img2_faces_cropped
    faces_cropped = list(
        map(lambda x: numpy_to_model_format(x, add_batch_dim=True), faces_cropped)
    )

    model = restore_model(model_path, device="cpu")
    model.eval()

    faces_mapped = [
        (i, faces_cropped[i], model(faces_cropped[i]))
        for i in range(len(faces_cropped))
    ]

    # Store the diffs to sort them and display by similarity
    diffs = []

    for face_i in range(len(faces_mapped)):
        rest = [faces_mapped[j] for j in range(face_i + 1, len(faces_mapped))]
        face1_idx, face1_cropped, face1_embed = faces_mapped[face_i]

        face_image_numpy = model_format_to_numpy(face1_cropped)

        # Write the image to be able to identify the faces individually
        cv2.imwrite(
            f"Face_{face1_idx}.jpg", cv2.cvtColor(face_image_numpy, cv2.COLOR_RGB2BGR)
        )

        for face2 in rest:
            face2_idx, _, face2_embed = face2
            face_diff = torch.sum(torch.abs(face1_embed - face2_embed))
            diffs.append((face1_idx, face2_idx, face_diff))
    
    diffs_sorted = list(sorted(diffs, key=lambda x : x[2]))

    for f1, f2, diff in diffs_sorted:
        print(f"Difference of Face {f1} and {f2} = {diff}")
        


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
    parser.add_argument("-t", "--threshold", action="store", dest="threshold", default=1.0, type=float)
    parser.add_argument("-f", "--frames-to-skip", action="store", dest="frames_to_skip", default=30, type=int)
    args = parser.parse_args()

    if args.video_path is None and args.img1 is None:
        raise ValueError("Missing arguments ...")

    elif args.video_path is not None:
        run_inference_video(args.model_path, args.video_path, each_nth_frame=args.frames_to_skip, unique_dist_threshold=args.threshold)

    elif args.img1 is not None and args.img2 is not None:
        run_inference_images(args.model_path, args.img1, args.img2)

    else:
        print("Unrecognized combination of arguments, exitting ...")
