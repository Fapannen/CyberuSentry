from torch.utils.data import Sampler


class IdentitySampler(Sampler):
    """IdentitySampler that samples more samples of a single
    class from the dataset to make sure there are enough
    positive samples for the triplet loss.

    The class creates a generator that yields the indices
    into a torch.nn.Dataset.__getitem__() method for each
    batch.

    This samples works as follows. Say the batch size is 8
    and we want at least 3 samples per id in the batch.
    (Usually its for the best to pick a number X that makes
    batch size % X == 0, but it should work on weird numbers
    as well.). To ensure 3 samples per id in the batch,
    the generator calculates how many ids will fit into the
    batch. In this case, 8 // 3 = 2, so only samples of 2
    classes will be included here. The result may look sth
    like [0, 1, 0, 1, 0, 1, 0, 1]
    """

    def __init__(self, num_identities: int, batch_size: int, min_samples_per_id: int):
        """Init the Sampler

        Parameters
        ----------
        num_identities : int
            How many identities (and thus mapping keys)
            are there in the sampled dataset. This sampler
            yields the indicies so it needs to be aware
            until which index to generate them.
        batch_size : int
            Target batch size
        min_samples_per_id : int
            Number of samples per identity that must be
            guaranteed to be in the final batch. Sampler
            ensures that by yielding some indices multiple
            times.
        """
        self.num_identities = num_identities

        self.batch_size = batch_size

        # At least how many samples should be present
        # in a single batch
        self.min_samples_per_id = min_samples_per_id

        # Otherwise it doesnt make sense
        assert self.batch_size >= self.min_samples_per_id

        # How many ids will fit into the batch given that
        # at least X must always be there
        self.ids_per_batch = self.batch_size // self.min_samples_per_id

        self.num_batches = self.num_identities // self.ids_per_batch

    def __iter__(self):
        for i in range(self.num_batches):
            # ie. (          [0, 1]              ) * 4 --> [0, 1, 0, 1, 0, 1, 0, 1]
            ids = (list(range(self.ids_per_batch)) * (self.min_samples_per_id + 1))[
                : self.batch_size
            ]

            # ie. [0, 1, 0, 1, 0, 1, 0, 1] in first epoch and
            #     [2, 3, 2, 3, 2, 3, 2, 3] in the second epoch
            # and so on, until the dataset is exhausted
            yield [(i * self.ids_per_batch) + j for j in ids]

    def __len__(self):
        return self.num_batches
