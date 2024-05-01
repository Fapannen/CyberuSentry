import numpy as np


def crop_bbox_from_image(
    img: np.ndarray,
    xmin: float | int,
    xmax: float | int,
    ymin: float | int,
    ymax: float | int,
) -> np.ndarray:
    """Crop a bounding box from image, as specified by the xmin, xmax, ymin, ymax
    parameters

    Parameters
    ----------
    img : np.ndarray
        Image from which bbox is to be cropped
    xmin : int
        min x coordinate value of the bbox
    xmax : int
        max x coordinate value of the bbox
    ymin : int
        min y coordinate value of the bbox
    ymax : int
        max y coordinate value of the bbox
    """
    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

    xmin = xmin if xmin >= 0 else 0

    ymin = ymin if ymin >= 0 else 0

    xmax = xmax if xmax < img.shape[1] else img.shape[1] - 1

    ymax = ymax if ymax < img.shape[1] else img.shape[0] - 1

    return img[ymin:ymax, xmin:xmax, :]
