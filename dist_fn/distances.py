import torch

"""
Implementation of distance functions used throughout
this project. When working with these distances,
keep in mind that in default settings, these classes
will return a tensor which still tracks grads. During
inference, it might be a good idea to get just the
'.item()' value.
"""


class EuclideanDistance:
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the euclidean distance between two tensors.
        Tensors are expected to be either the same, or
        broadcastable shapes, no additional handling is done.

        Returns
        -------
        torch.Tensor
            Tensor containing the computed distance(s).
            Don't forget to use '.item()' during inference,
            otherwise gradients will be stored.
        """
        return torch.sum(torch.square(torch.abs(a - b)))


class CosineDistance:
    def __init__(self) -> None:
        self.fn = torch.nn.CosineSimilarity()

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the Cosine distance between two tensors.
        torch.nn.CosineSimilarity() cannot be applied on
        1-D tensors, so any 1-D tensor passed to this
        function will implicitely get unsqueezed to conform
        to this function's requirements.

        Since

        CosineDistance(a, b) = 1 - CosineSimilarity(a, b)

        keep in mind that torch.nn.CosineSimilarity() returns
        values between [-1, 1], not [0, 1]. This function
        can thus return values in range [0, 2] !

        Returns
        -------
        torch.Tensor
            Computed distance between tensors 'a' and 'b'.
            Don't forget to use '.item()' during inference,
            otherwise gradients will be stored.
        """
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        # Distance = 1 - similarity
        return 1.0 - self.fn(a, b)
