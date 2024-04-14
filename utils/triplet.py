import torch
from typing import Literal


def dist(a, b):
    return torch.sum(torch.square(torch.abs(a - b))).item()


def build_triplets(
    pos_embs: torch.Tensor,
    neg_embs: torch.Tensor,
    min_samples_per_id: int,
    triplet_setting=Literal["semi-hard", "hard"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build triplets for each positive sample
    in pos_embs. It is expected that pos_embs
    contains at least cfg.min_samples_per_id
    samples per identity for optimal learning.

    Parameters
    ----------
    pos_embs : torch.Tensor
        Embeddings of the positive samples. Note that
        there should be multiple images per identity
        in the batch, and that the batch does not need
        to contain just a single identity.

        shape [B, embed_dim]

    neg_embs : torch.Tensor
        Embeddings of the negative samples. Keep in mind
        that the negative samples are only negative to
        the identity within the sample! As in the negative
        for index i might not be negative at all for index j.

        shape [B, embed_dim]

    min_samples_per_id : int
        Number of minimal samples per identity that
        are to be present within a batch.

    triplet_setting : Literal["semi-hard", "hard"]
        The strategy of generating the triplets.

        # TODO explain with links


    Returns
    -------
    List[torch.Tensor, torch.Tensor, torch.Tensor]
        shape [3, num_semi_hard_samples, embed_dim]
    """
    triplets = []

    # Length of one cycle, basically the number of samples
    # before the same identity is reached again
    cycle_length = len(pos_embs) // min_samples_per_id

    for i in range(len(pos_embs)):
        # Get indices of embeddings of the same identity in the batch
        # as "looking forward", ie. the positives already processed
        # will not be there to avoid redundancy
        other_positives_indices = list(range(i, len(pos_embs), cycle_length))[1:]
        if len(other_positives_indices) == 0:
            continue

        # Gather embeddings from the indicies
        index = torch.tensor(other_positives_indices).cuda()
        other_positives_embeds = torch.index_select(pos_embs, dim=0, index=index)

        # Determine the index of the hardest positive - most distant from the
        # original one while still being the same identity. Note that the
        # index is index to "other_positive_indices", not pos_embs yet!
        hardest_positive_idx = torch.argmax(
            torch.sum(torch.abs(other_positives_embeds - pos_embs[i]), dim=1)
        ).item()

        # Save the hardest positive
        hardest_embedding = pos_embs[other_positives_indices[hardest_positive_idx]]

        if triplet_setting == "semi-hard":
            indices_satisfies_semihard = []
            # list(all negatives with regards to the base identity)
            for k in list(range(i % cycle_length, len(pos_embs), cycle_length)):
                if dist(pos_embs[i], hardest_embedding) < dist(
                    pos_embs[i], neg_embs[k]
                ):
                    indices_satisfies_semihard.append(k)
                    triplet = torch.stack((pos_embs[i], hardest_embedding, neg_embs[k]))
                    triplets.append(triplet)

        if triplet_setting == "hard" or len(indices_satisfies_semihard) == 0:
            # Pick the hardest (here argmin because the smaller the distance from negative, the harder it is)

            # We need to pick the negatives from the indices on which samples of the same identity lie.
            # If we have a sample of identity X on indices 'YZ', we know that all negatives on indices
            # YZ are also negative to all other samples of identity X. We cannot however pick from
            # any negative index, because there, for identity Y, an identity X might have been chosen
            # as negative, which would confuse the network.
            negatives_indices = list(
                range(i % cycle_length, len(pos_embs), cycle_length)
            )

            # Get the negative samples with regards to identity i
            negatives_to_identity = torch.index_select(
                neg_embs, dim=0, index=torch.tensor(negatives_indices).cuda()
            )

            # Determine which of the negatives is the hardest. The index is however
            # to "negatives_indices", not to neg_embs yet!
            hardest_negative_idx = torch.argmin(
                torch.sum(torch.abs(negatives_to_identity - pos_embs[i]), dim=1)
            ).item()

            triplet = torch.stack(
                (
                    pos_embs[i],
                    hardest_embedding,
                    neg_embs[negatives_indices[hardest_negative_idx]],
                )
            )
            triplets.append(triplet)

    return torch.stack(triplets, dim=1)
