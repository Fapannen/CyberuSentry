import timm
import torch.nn as nn


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=128) -> None:
        super().__init__()
        self.model = timm.create_model(
            "mobilenetv2_100.ra_in1k", pretrained=True, num_classes=num_classes
        )
        self.act = nn.Tanh()

    def forward(self, x):
        # Bound values to (-1, 1), pure outputs from
        # the model are unbounded, which can be
        # unstable when computing distances.

        return self.act(self.model(x))
