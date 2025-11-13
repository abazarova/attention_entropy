from sklearn.model_selection import KFold, train_test_split


class KFold3(KFold):
    '''Split dataset X into train, val and test subsets.

    Args:
        X (array-like): Training data.

    Returns:
        train_indxs (ndarray): The training set indices for that split.
        val_indxs (ndarray): The validation set indices for that split.
        test_indxs (ndarray): The test set indices for that split.

    '''
    def split(self, X):
        s = super().split(X)
        for train_indxs, test_indxs in s:
            train_indxs, val_indxs = train_test_split(
                train_indxs, test_size=len(test_indxs), random_state=self.random_state
            )
            yield train_indxs, val_indxs, test_indxs
