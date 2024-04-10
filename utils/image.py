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

        Parameters
        ----------
        path : str | Path
                Full path to the image to be read.
        convert_to_tensor : bool, optional
                Convert the image to PyTorch tensor
                (Including the HWC -> CHW conversion)
        scale : bool, optional
                Scale the values into [0, 1] range


        Returns
        -------
        np.ndarray | torch.Tensor
                Opened image either as a
                {np.ndarray, RGB, HWC}
                or
                {torch.Tensor, RGB, CHW}
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if scale:
        image = image / 255.0

    image = numpy_to_model_format(image) if convert_to_tensor else image

    return image


def numpy_to_model_format(
    image: np.ndarray, add_batch_dim: bool = False
) -> torch.Tensor:
    """Convert a numpy image to a torch Tensor.
            Includes:
        - Scaling to [0, 1] if the passed image is not yet scaled
        - Resize to the model's input size
        - Conversion from ndarray to Tensor
        - HWC -> CHW
        - Conversion to float
        - Optional adding of batch dimension

    Parameters
    ----------
    image : np.ndarray
            Image as a numpy array. Expects to already be
            in BGR and HWC format.
    add_batch_dim : bool, optional
            _description_, by default False

    Returns
    -------
    torch.Tensor
            processed image in Tensor format, ready to
            be consumed by the model.
    """
    input_size = (256, 256)

    image = cv2.resize(image, input_size)

    if np.max(image) > 1:
        image = image / 255.0

    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).float()

    if add_batch_dim:
        image = image.unsqueeze(0)

    return image


def model_format_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert Tensor represented image to numpy representation

    Parameters
    ----------
    image : torch.Tensor
            Image in Tensor format. Can be batched.

    Returns
    -------
    np.ndarray
            Image in np.ndarray format in HWC, uint8
    """
    # Squeeze batch dimension. If there is none, this will do
    # nothing
    image = image.squeeze(0)

    image = image.permute(1, 2, 0).numpy() * 255
    image = image.astype(np.uint8)
    return image
