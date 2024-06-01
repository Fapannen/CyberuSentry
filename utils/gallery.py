import torch
from typing import Any

from dist_fn.distances import EuclideanDistance


def gallery_similarity(
    gallery: dict[Any, torch.Tensor],
    face_embedding: torch.Tensor,
    dist_fn: str,
    eucl_dist_thr: float = None,
    w: float = None,
) -> torch.Tensor:
    """
    Take a given 'face_embedding' and match it against embeddings
    in 'gallery'. Return the computed distances of the face_embed
    compared to all subjects in the gallery.

    In case of euclidean distance as the distance function, the
    distances are first max-normalized into [0, 1] range and then
    the "matching" subjects are selected depending on a threshold.
    Subjects with distance lower than threshold are considered matching,
    subjects with distance larger than threshold are considered non-matching.

    Output from this function is a torch.Tensor containing the distances
    of the provided face embed to all face embeddings in the gallery and
    the values are in the [0, 1] range.

    Parameters
    ----------
    gallery : dict[Any, torch.Tensor]
        A dictionary of identities. For each identity in the gallery, there
        is the identity's embedding(s) stored, which is compared to the provided
        face embedding. The key in the gallery can be anything, as it is not
        used at all when computing the similarity.
    face_embedding : torch.Tensor
        The embedding from an inferred image. This is the vector that represents
        the face from whatever image to be matched against known identities in the
        gallery.
    dist_fn
        Method of computing the distance between face embeddings. Can be either
        "cosine" or "euclidean".
    eucl_dist_thr
        Used in the case of euclidean distance function. Determines the threshold
        under which the distances are considered matching
    w
        Controls the value at which the non-matching samples scores start

    Returns
    -------
    torch.Tensor
        Tensor of distances of the embedding to all subjects in the gallery
    """
    gallery_embeddings = [gallery[subject] for subject in gallery]
    gallery_embeddings = torch.stack(gallery_embeddings)

    metric_fn = (
        EuclideanDistance() if dist_fn == "euclidean" else torch.nn.CosineSimilarity()
    )

    gallery_similarities = torch.tensor(
        [
            metric_fn(subject_embedding, face_embedding).detach().item()
            for subject_embedding in gallery_embeddings
        ],
        requires_grad=False,
    )

    if dist_fn == "euclidean":
        max_dist = torch.max(gallery_similarities).item()
        gallery_similarities /= max_dist
        threshold = eucl_dist_thr / max_dist if eucl_dist_thr > 1.0 else eucl_dist_thr
        gallery_similarities = torch.tensor(
            [
                max(1 / (1 + score), 0.75) if score <= threshold else max(w - score, 0.0)
                for score in gallery_similarities.numpy()
            ],
            requires_grad=False,
        )
    
    if dist_fn == "cosine":
        # CosineSimilarity outputs values in [-1, 1] range
        # so shift them to [0, 1].
        gallery_similarities = (gallery_similarities + 1.0) / 2
        
    assert torch.min(gallery_similarities) >= 0.0 and torch.max(gallery_similarities) <= 1.0

    # It is actually gallery similarity, not dist
    return gallery_similarities