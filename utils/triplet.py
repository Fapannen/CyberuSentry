import torch
from typing import Literal


def dist(a, b):
    return torch.sum(torch.square(torch.abs(a - b))).item()


def build_triplets(
    pos_embs: torch.Tensor,
    neg_embs: torch.Tensor,
    min_samples_per_id: int,
    triplet_setting=Literal["semi-hard", "hard", "batch-hard"],
    margin: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build triplets
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

    triplet_setting : Literal["semi-hard", "hard", "batch-hard"]
        The strategy of generating the triplets.

        - Semi-hard:  For each anchor-positive pair, select
                      the hardest negative, which satisfies the
                      following equation

                      |d(a, p) - d(a, n)| < margin
                      &&
                      d(a, p) < d(a, n)

        - hard:       For each anchor-positive pair, select
                      the hardest negative, ie the negative
                      sample which is the closest to the
                      anchor

        - batch-hard: For each positive sample, select the
                      hardest positive and the hardest
                      negative.
                      (This option is kinda "ultra-hard")

    margin: float
        The margin imposed between embeddings of classes.
        Required only for the "semi-hard" option.

    Returns
    -------
    List[torch.Tensor, torch.Tensor, torch.Tensor]
        shape [3, <computed_samples>, embed_dim]

        The number of computed samples may vary.
    """

    # batch-hard requires a little different logic
    if triplet_setting == "batch-hard":
        return build_batch_hard_triplets(pos_embs, neg_embs, min_samples_per_id)

    triplets = []

    # Length of one cycle. The number of samples
    # before the same identity is reached again
    cycle_length = len(pos_embs) // min_samples_per_id

    for i in range(len(pos_embs)):
        # Get indices of all other samples belonging to the same class
        other_positives_indices = list(
            range(i % cycle_length, len(pos_embs), cycle_length)
        )
        other_positives_indices.remove(i)  # Dont compare with itself

        # As said in the paper:
        # Instead of picking the hardest positive, we use
        # *ALL ANCHOR-POSITIVE PAIRS* in a mini-batch while
        # still selecting the hard negatives
        for j in other_positives_indices:
            pos_emb1 = pos_embs[i]
            pos_emb2 = pos_embs[j]

            if triplet_setting == "semi-hard":
                indices_satisfies_semihard = []

                for k in other_positives_indices:
                    # If the positive and negative are ordered correcly (pos is closer than neg)
                    # AND
                    # the difference of distances (pos1, pos2) and (pos1, neg) lies in the margin
                    # THEN
                    # This combination is semi-hard because it is correctly ordered, but we wanna
                    # push them apart a little.
                    if (dist(pos_emb1, pos_emb2) < dist(pos_emb1, neg_embs[k])) and abs(
                        dist(pos_emb1, neg_embs[k]) - dist(pos_emb1, pos_emb2)
                    ) < margin:
                        indices_satisfies_semihard.append(k)

                # Select the hardest negative, which still satisfies the semi-hard condition
                # Hardest = closest to the anchor
                if len(indices_satisfies_semihard) != 0:
                    semi_hard_neg_embeds = torch.index_select(
                        neg_embs,
                        dim=0,
                        index=torch.tensor(indices_satisfies_semihard).cuda(),
                    )
                    hardest_satisfying_negative = torch.argmin(
                        torch.sum(torch.abs(semi_hard_neg_embeds - pos_embs[i]))
                    )
                    triplet = torch.stack(
                        (
                            pos_emb1,
                            pos_emb2,
                            neg_embs[
                                indices_satisfies_semihard[hardest_satisfying_negative]
                            ],
                        )
                    )
                    triplets.append(triplet)

            if triplet_setting == "hard" or len(indices_satisfies_semihard) == 0:
                # Pick the hardest negative (here argmin because the smaller the distance from negative, the harder it is)

                # We need to pick the negatives from the indices on which samples of the same identity lie.
                # If we have a sample of identity X on indices 'YZ', we know that all negatives on indices
                # YZ are also negative to all other samples of identity X. We cannot however pick from
                # any negative index, because there, for identity Y, an identity X might have been chosen
                # as negative, which would confuse the network.
                negatives_indices = other_positives_indices

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
                        pos_emb1,
                        pos_emb2,
                        neg_embs[negatives_indices[hardest_negative_idx]],
                    )
                )
                triplets.append(triplet)

    return torch.stack(triplets, dim=1)


def build_batch_hard_triplets(
    pos_embs: torch.Tensor,
    neg_embs: torch.Tensor,
    min_samples_per_id: int,
    margin: int = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    triplets = []

    # For explanatory comments, see the original function above.
    cycle_length = len(pos_embs) // min_samples_per_id

    for i in range(len(pos_embs)):
        other_positives_indices = other_positives_indices = list(
            range(i % cycle_length, len(pos_embs), cycle_length)
        )

        other_positives_indices.remove(i)  # Do not compare with itself :)

        index = torch.tensor(other_positives_indices).cuda()
        other_positives_embeds = torch.index_select(pos_embs, dim=0, index=index)

        # Determine the index of the hardest positive
        hardest_positive_idx = torch.argmax(
            torch.sum(torch.abs(other_positives_embeds - pos_embs[i]), dim=1)
        ).item()

        # Save the hardest positive
        hardest_positive_embedding = pos_embs[
            other_positives_indices[hardest_positive_idx]
        ]

        # Now pick the hardest negative
        negatives_indices = other_positives_indices

        # Get the negative samples with regards to identity i
        negatives_to_identity = torch.index_select(
            neg_embs, dim=0, index=torch.tensor(negatives_indices).cuda()
        )

        # Determine which of the negatives is the hardest.
        hardest_negative_idx = torch.argmin(
            torch.sum(torch.abs(negatives_to_identity - pos_embs[i]), dim=1)
        ).item()

        triplet = torch.stack(
            (
                pos_embs[i],
                hardest_positive_embedding,
                neg_embs[negatives_indices[hardest_negative_idx]],
            )
        )
        triplets.append(triplet)

    return torch.stack(triplets, dim=1)
