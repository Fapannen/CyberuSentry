import facenet_pytorch as fp
import torch
import argparse
import cv2
import numpy as np

from utils.image import (
    read_image,
    numpy_to_model_format,
    model_format_to_numpy
) 
from utils.model import restore_model


def run_inference_video():
    pass


def get_cropped_faces(detected_faces, image):
    """# Output from fp model is a tuple (bbox, confidence)
    # ie. ([[tl_x, tl_y, br_x, br_y], [F2_tl_x, ...]], [[0.99], [0.45]])
    # Dont forget that image is in shape (height, width)

    Args:
        detected_faces (_type_): tuple [bboxes, confidences]
        image (_type_): The original image from which to cut faces

    Returns:
        List of images: List of images representing the cut faces.
                        The images are preprocessed to be properly
                        fed into the encoder network.
    """
    cropped_faces = []
    for face_bbox in detected_faces[0]:
        print(face_bbox)
        xmin, ymin, xmax, ymax = face_bbox

        bb_xdiff = xmax - xmin
        bb_ydiff = ymax - ymin

        new_bb_xmin = int(xmin - (bb_xdiff // 4))
        new_bb_xmin = new_bb_xmin if new_bb_xmin >= 0 else 0

        new_bb_ymin = int(ymin - (bb_ydiff // 4))
        new_bb_ymin = new_bb_ymin if new_bb_ymin >= 0 else 0

        new_bb_xmax = int(xmax + (bb_xdiff // 4))
        new_bb_xmax = (
            new_bb_xmax if new_bb_xmax < image.shape[1] else image.shape[1] - 1
        )

        new_bb_ymax = int(ymax + (bb_ydiff // 4))
        new_bb_ymax = (
            new_bb_ymax if new_bb_ymax < image.shape[1] else image.shape[0] - 1
        )

        crop = image[new_bb_ymin:new_bb_ymax, new_bb_xmin:new_bb_xmax, :]

        cropped_faces.append(crop)

    return cropped_faces


def run_inference_images(model_path: str, img1: str, img2: str):
    """Run inference on two images.
    # TODO: Decide whether this function should work as is now or
    whether it should output the number of unique faces ...

    Args:
        model_path (str): Path to the model checkpoint
        img1 (str): Path to the first image
        img2 (str): Path to the second image
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
            face_diff = torch.sum(face1_embed - face2_embed)
            cos_sim = torch.nn.CosineSimilarity()
            face_cossim = (cos_sim(face1_embed, face2_embed)).item()
            print(
                f"The difference of face {face1_idx} versus face {face2_idx} is {face_diff}"
                f" and their cosine similarity is {face_cossim}"
            )


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
    args = parser.parse_args()

    if args.video_path is None and args.img1 is None:
        raise ValueError("Missing arguments ...")

    elif args.video_path is not None:
        run_inference_video()

    elif args.img1 is not None and args.img2 is not None:
        run_inference_images(args.model_path, args.img1, args.img2)

    else:
        print("Unrecognized combination of arguments, exitting ...")
