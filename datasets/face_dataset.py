import torch
import torchvision
import os
import random

from pathlib import Path
from typing import Literal
from torch.utils.data import Dataset
from utils.image import read_image


class FaceDataset(Dataset):
    """General Face Dataset for this repository.
    Takes care of preparing training tuples for triplet loss etc.

    Datasets derived from this class should only override a method
    desired to establish {id:[img1, img2]} mapping from id to a set
    of images of that 'id' identity.
    """

    TBD AT HOME