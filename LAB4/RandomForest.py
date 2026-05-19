import numpy as np
from DecisionTree import (DecisionTree)

class RandomForest:

    def __init__(
        self,
        n_trees=10,
        max_depth=5
    ):

        self.n_trees = n_trees

        self.max_depth = max_depth

        self.trees = []

    # BOOTSTRAP SAMPLING
    def bootstrap_sample(
        self,
        X,
        y
    ):

        n_samples = len(X)

        indices = np.random.choice(
            n_samples,
            n_samples,
            replace=True
        )

        X_sample = X[indices]

        y_sample = y[indices]

        return (
            X_sample,
            y_sample
        )

    # TRAIN
    def fit(
        self,
        X,
        y
    ):

        self.trees = []

        for _ in range(
            self.n_trees
        ):

            (
                X_sample,
                y_sample
            ) = self.bootstrap_sample(
                X,
                y
            )

            tree = DecisionTree(
                max_depth=self.max_depth
            )

            tree.fit(
                X_sample,
                y_sample
            )

            self.trees.append(tree)

    # MAJORITY VOTING
    def predict(
        self,
        X
    ):

        tree_predictions = []

        for tree in self.trees:

            predictions = tree.predict(X)

            tree_predictions.append(
                predictions
            )

        tree_predictions = np.array(
            tree_predictions
        )
        
        tree_predictions = np.swapaxes(
            tree_predictions,
            0,
            1
        )

        final_predictions = []

        for predictions in tree_predictions:

            values, counts = np.unique(
                predictions,
                return_counts=True
            )

            majority_vote = values[
                np.argmax(counts)
            ]

            final_predictions.append(
                majority_vote
            )

        return np.array(
            final_predictions
        )