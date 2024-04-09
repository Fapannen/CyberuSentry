import facenet_pytorch as fp
import cv2
from pathlib import Path
import os
import argparse

from tqdm import tqdm

from utils.image import read_image
from inference import get_cropped_faces


def crop_images_in_folder(src_path, dst_path):
    """Crop the CelebA dataset so that the images contain just the face
    and a little of surroundings, but not as much as the original images.
    This should enable the model to focus more on the faces rather than
    perhaps surroundings.

    The images retain their original name.
    """
    path_to_dataset = src_path
    output_path = Path(dst_path)
    output_path.mkdir(exist_ok=True, parents=True)

    fp_model = fp.MTCNN()

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
    args = parser.parse_args()

    if args.src_path is None or args.dst_path is None:
        raise ValueError("Missing arguments ...")

    crop_images_in_folder(args.src_path, args.dst_path)
