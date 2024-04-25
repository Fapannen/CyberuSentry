import facenet_pytorch as fp
import cv2
from pathlib import Path
import os
import argparse

from typing import Literal

from tqdm import tqdm

from utils.image import read_image
from inference import get_cropped_faces


def crop_images_in_folder(src_path: str, dst_path: str, type: Literal["v1", "v1"]):
    """Crop the provided dataset so that the images contain just the face
    and a little of surroundings, but not as much as the original images.
    This should enable the model to focus more on the faces rather than
    perhaps surroundings.

    The images retain their original name.

    Parameters
    ----------
    src_path : str | Path
        Path to the directory containing images to be cropped
    dst_path : str | Path
        Path to the directory where the results shall be saved
    type
        Type of processing. Valid values are "v1" and "v2"

        v1:
            Was made for CelebA dataset where all images are in
            a single folder and annotations are provided in an
            external file.

            path_to_dataset/
                - img1
                - img2
                - img3
                - ...

        v2:
            Was made for LFW and similar datasets, where the
            dataset dir contains per-identity directories where
            the target images reside.

            path_to_dataset/
                - identity_dir1
                    - img1
                    - img2
                - identity_dir2
                    - img1
                - ...

    """
    path_to_dataset = src_path
    output_path = Path(dst_path)
    output_path.mkdir(exist_ok=True, parents=True)

    fp_model = fp.MTCNN()

    # V1 = The whole directory contains only images to be cropped
    if type == "v1":
        for image_path in tqdm(os.listdir(path_to_dataset)):
            image_path_full = path_to_dataset + "/" + image_path
            image = read_image(image_path_full, convert_to_tensor=False, scale=False)
            img_face = fp_model.detect(image)
            if img_face[0] is None:
                print(f"Image {image_path} has no face detected, saving as is ...")
                face_cropped = image
            else:
                face_cropped = get_cropped_faces(img_face, image)[0]
            cv2.imwrite(
                str(output_path) + "/" + image_path,
                cv2.cvtColor(face_cropped, cv2.COLOR_RGB2BGR),
            )

    # V2 = The directory contains per-identity subdirectories
    elif type == "v2":
        for identity_dir in os.listdir(path_to_dataset):
            identity_dir_full_path = f"{path_to_dataset}/{identity_dir}"
            crop_images_in_folder(
                identity_dir_full_path, f"{output_path}/{identity_dir}", "v1"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "-s",
        "--src-path",
        action="store",
        dest="src_path",
        help="Path to the directory containing images to be cropped",
    )
    parser.add_argument(
        "-d",
        "--dst-path",
        action="store",
        dest="dst_path",
        help="Path to the directory where the cropped images are to be stored",
    )
    parser.add_argument(
        "-t",
        "--type",
        action="store",
        dest="type",
        help="Which version of crop_dataset to use. V1 assumes the directory contains only images, V2 assumes the directory contains identity directories",
    )
    args = parser.parse_args()

    if args.src_path is None or args.dst_path is None or args.type is None:
        raise ValueError("Missing arguments ...")

    crop_images_in_folder(args.src_path, args.dst_path, args.type)
