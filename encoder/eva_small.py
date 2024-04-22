import timm
import torch.nn as nn


class EvaSmall(nn.Module):
    def __init__(self, num_classes=128) -> None:
        super().__init__()
        self.model = timm.create_model(
            "eva02_base_patch14_224.mim_in22k",
            pretrained=False,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)
