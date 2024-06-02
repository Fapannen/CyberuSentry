import cv2
import torch
import numpy as np
from pathlib import Path


def read_image(path: str, scale: bool = True) -> np.ndarray:
    """Reads an image, converts it to RGB and optionally converts it to
    torch.Tensor or scales the values into [0, 1].

    This function does not do any resizing or input transformations
    (except scaling the values, if set) nor augmentations etc.
    These shall be handled elsewhere.

        Parameters
        ----------
        path : str | Path
                Full path to the image to be read.
        scale : bool, optional
                Scale the values into [0, 1] range

        Returns
        -------
        np.ndarray
                Opened image as a
                {np.ndarray, RGB, HWC}
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if scale:
        image = image / 255.0

    return image


def numpy_to_model_format(
    image: np.ndarray,
    target_size: int | tuple[int, int] = 224,
    add_batch_dim: bool = False,
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
    input_size :
            size of the model format input image. Can be
            either a single integer or a tuple.
    add_batch_dim : bool, optional
            Whether to produce a BCHW tensor or just a CHW.
            By default False -> CHW

    Returns
    -------
    torch.Tensor
            Processed image in Tensor format. If batch dim
            is added, the output should be directly
            consumable by the model.
    """
    input_size = (
        (target_size, target_size) if isinstance(target_size, int) else target_size
    )

    image = cv2.resize(image, input_size)

    if np.max(image) > 1:
        image = image / 255.0

    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).float()

    if add_batch_dim:
        image = image.unsqueeze(0)

    return image


def model_format_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert Tensor represented image to numpy representation.
    Ditch the batch dimension, change from CHW to HWC.

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


def save_batch_samples(batch: torch.Tensor, output_dir: str = "./debug") -> None:
    """Save a batch of images during training for
    manual examination if everything is working
    properly.

    Parameters
    ----------
    batch : torch.Tensor
        One batch of data, expected shape is BCHW
    output_dir : str, optional
        Path to a directory where the images shall be saved.
        By default "./debug"
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(exist_ok=True, parents=True)

    positives, negatives = batch

    for i in range(len(positives)):
        pos_img = positives[i]
        neg_img = negatives[i]

        cv2.imwrite(
            f"{str(output_dir)}/{str(i)}-pos.jpg",
            cv2.cvtColor(model_format_to_numpy(pos_img), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            f"{str(output_dir)}/{str(i)}-neg.jpg",
            cv2.cvtColor(model_format_to_numpy(neg_img), cv2.COLOR_RGB2BGR),
        )
