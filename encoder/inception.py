import torch.nn as nn
from torchvision.models import inception_v3


class SentryEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        self.model = inception_v3(num_classes=embedding_dim, weights=None)

    def forward(self, x):
        """Forward the inputs

        Parameters
        ----------
        x : torch.Tensor
            input to be forwarded

        Returns
        -------
        output
            Processed input
        """

        return self.model(x)
