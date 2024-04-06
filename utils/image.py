import cv2
import torch
import numpy as np

def read_image(path, convert_to_tensor : bool = True) -> np.ndarray | torch.Tensor:
	"""
	Reads an image, converts it to RGB and optionally converts it to 
	torch.Tensor.

	Note that this function does not do any resizing or input transformations,
	nor augmentations. These shall be handled later in the code.
	"""
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image / 255.0

	image = torch.from_numpy(image).permute(2, 0, 1).float() if convert_to_tensor else image

	return image