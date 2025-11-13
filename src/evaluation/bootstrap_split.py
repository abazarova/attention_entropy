import numpy as np
from sklearn.model_selection import BaseCrossValidator


class BootstrapSplitter(BaseCrossValidator):
    def __init__(
        self, n_splits=10, idxs_to_drop=None, train_size=None, random_state=None
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.random_state = random_state
        self.idxs_to_drop = idxs_to_drop

    def split(self, X, y=None, groups=None):
        if self.idxs_to_drop is not None:
            samples = np.setdiff1d(
                np.arange(len(X)), self.idxs_to_drop, assume_unique=True
            )
        else:
            samples = np.arange(len(X))

        n_samples = len(samples)
        np.random.seed(self.random_state)

        for _ in range(self.n_splits):
            # Randomly sample with replacement for the train set
            # train_indices = np.random.choice(samples, size=n_samples, replace=False)
            train_indices = np.random.choice(samples, size=n_samples, replace=True)
            # The validation set consists of the out-of-bag samples (those not in the train set)
            # test_indices = train_indices
            test_indices = np.setdiff1d(samples, train_indices, assume_unique=False)
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
