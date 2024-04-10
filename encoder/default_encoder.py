import torch.nn as nn


class SentryEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)

        self.maxpool = nn.MaxPool2d(3)
        self.activation = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.LazyLinear(embedding_dim)

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

        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.activation(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
