import torch


class EuclideanDistance:
    def __call__(self, a, b):
        return torch.sum(torch.square(torch.abs(a - b)))


class CosineDistance:
    def __call__(self, a, b):
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        if len(b.shape) == 1:
            b = b.unsqueeze(0)
            
        return torch.nn.CosineSimilarity()(a, b)
