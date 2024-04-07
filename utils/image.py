import cv2
import torch
import numpy as np


def read_image(
    path, convert_to_tensor: bool = True, scale: bool = True
) -> np.ndarray | torch.Tensor:
	"""Reads an image, converts it to RGB and optionally converts it to
    torch.Tensor or scales the values into [0, 1].

    Note that this function does not do any resizing or input transformations,
    nor augmentations. These shall be handled later in the code.

	Args:
		path (str): Full path to the image to be read.
		convert_to_tensor (bool, optional): Convert the image to PyTorch tensor
											(Including the HWC -> CHW conversion)
		scale (bool, optional): Scale the values into [0, 1] range

	Returns:
		np.ndarray | torch.Tensor: Opened image either as a
									{np.ndarray, RGB, HWC}
                                    or
                                    {torch.Tensor, RGB, CHW}
	"""
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	if scale:
		image = image / 255.0

	image = (
        torch.from_numpy(image).permute(2, 0, 1).float() if convert_to_tensor else image
    )

	return image
