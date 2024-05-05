import timm
import torch.nn as nn


class EvaTiny(nn.Module):
    def __init__(self, num_classes=128) -> None:
        super().__init__()
        self.model = timm.create_model(
            "eva02_tiny_patch14_224.mim_in22k", pretrained=False
        )
        self.act = nn.Tanh()

    def forward(self, x):
        # Bound values to (-1, 1), pure outputs from
        # the model are unbounded, which can be
        # unstable when computing distances.
        
        return self.act(self.model(x))
