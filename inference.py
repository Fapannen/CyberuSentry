import facenet_pytorch as fp
import torch
import argparse
import cv2

from utils.image import read_image
from utils.model import restore_model

def run_inference_video():
    pass

def run_inference_images(model_path, img1, img2):
    img1 = read_image(img1, convert_to_tensor=False, scale=False)
    img2 = read_image(img2, convert_to_tensor=False, scale=False)

    fp_model = fp.MTCNN()

    img1_faces = fp_model.detect(img1)
    img2_faces = fp_model.detect(img2)

    if img1_faces[0] is None or img2_faces[0] is None:
        print("No faces detected in one of the images!")
        print("Exitting ...")
        return
    
    # Output from fp model is a tuple (bbox, confidence)
    # ie. ([[tl_x, tl_y, br_x, br_y], [F2_tl_x, ...]], [[0.99], [0.45]])
    # Dont forget that image is in shape (height, width)
    # TODO: Rewrite before publishing!!!
    faces_cropped = []
    for face in (img1_faces, img2_faces):
        for face_bbox in face[0]:
            tl_y, tl_x, br_y, br_x = face_bbox
            bb_height = abs(tl_x - br_x)
            bb_width = abs(tl_y - br_y)

            new_bb_tlx = int(tl_x - (bb_height // 2))
            new_bb_tlx = new_bb_tlx if new_bb_tlx >= 0 else 0

            new_bb_tly = int(tl_y - (bb_width // 2))
            new_bb_tly = new_bb_tly if new_bb_tly >= 0 else 0

            new_bb_brx = int(br_x + (bb_height // 2))
            new_bb_brx = new_bb_brx if new_bb_brx < img1.shape[0] else img1.shape[0] - 1

            new_bb_bry = int(br_y + (bb_width // 2))
            new_bb_bry = new_bb_bry if new_bb_bry < img1.shape[1] else img1.shape[1] - 1

            crop = img1[new_bb_tlx : new_bb_brx, new_bb_tly : new_bb_bry, :]
            
            # Prepare for feeding into the network
            preprocessed_crop = cv2.resize(crop, (256, 256))
            preprocessed_crop = preprocessed_crop / 255.0
            preprocessed_crop = torch.from_numpy(preprocessed_crop)
            preprocessed_crop = preprocessed_crop.permute(2, 0, 1).float()
            preprocessed_crop = preprocessed_crop.unsqueeze(0)

            faces_cropped.append(preprocessed_crop)

    model = restore_model(model_path, device="cpu")

    faces_mapped = [(i, faces_cropped[i], model(faces_cropped[i])) for i in range(len(faces_cropped))]

    for face1 in faces_mapped:
        rest = [face for face in faces_mapped if face != face1]
        face_idx, face_cropped, face_embed = face1
        cv2.imwrite(f"Face_{face_idx}.jpg", cv2.cvtColor(face_cropped.squeeze(0).permute(1, 2, 0).numpy() * 255, cv2.COLOR_RGB2BGR))
        for face2 in rest:
            print(f"The difference of face {face_idx} versus face {face2[0]} is {torch.sum(face_embed - face2[2])}")

        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-m", "--model-path", action="store", dest="model_path", default="best.ckpt")
    parser.add_argument("-i1", "--img1", action="store", dest="img1")
    parser.add_argument("-i2", "--img2", action="store", dest="img2")
    parser.add_argument("-v", "--video-path", action="store", dest="video_path", default="vid/sample_video.mp4")
    args = parser.parse_args()

    run_inference_images(args.model_path, args.img1, args.img2)
