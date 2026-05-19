import numpy as np

class Node:

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:

    def __init__(self, max_depth=5):

        self.max_depth = max_depth
        self.root = None


    def gini(self, y):

        classes = np.unique(y)

        impurity = 1

        for c in classes:

            p = np.sum(y == c) / len(y)

            impurity -= p ** 2

        return impurity


    def information_gain(
        self,
        y,
        y_left,
        y_right
    ):

        parent_gini = self.gini(y)

        n = len(y)

        n_left = len(y_left)
        n_right = len(y_right)

        weighted_gini = (
            (n_left / n) * self.gini(y_left)
            + (n_right / n) * self.gini(y_right)
        )

        return parent_gini - weighted_gini


    def split(
        self,
        X,
        y,
        feature,
        threshold
    ):

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        X_left = X[left_mask]
        y_left = y[left_mask]

        X_right = X[right_mask]
        y_right = y[right_mask]

        return (
            X_left,
            y_left,
            X_right,
            y_right
        )

    def best_split(self, X, y):

        best_gain = -1

        split_feature = None
        split_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):

            thresholds = np.unique(
                X[:, feature]
            )

            for threshold in thresholds:

                (
                    X_left,
                    y_left,
                    X_right,
                    y_right
                ) = self.split(
                    X,
                    y,
                    feature,
                    threshold
                )

                if (
                    len(y_left) == 0
                    or len(y_right) == 0
                ):
                    continue

                gain = self.information_gain(
                    y,
                    y_left,
                    y_right
                )

                if gain > best_gain:

                    best_gain = gain

                    split_feature = feature
                    split_threshold = threshold

        return split_feature, split_threshold


    def most_common_label(self, y):

        values, counts = np.unique(
            y,
            return_counts=True
        )

        return values[np.argmax(counts)]


    def grow_tree(
        self,
        X,
        y,
        depth=0
    ):

        n_samples = len(y)

        n_classes = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_classes == 1
            or n_samples == 0
        ):

            leaf_value = self.most_common_label(y)

            return Node(value=leaf_value)

        feature, threshold = self.best_split(
            X,
            y
        )

        if feature is None:

            leaf_value = self.most_common_label(y)

            return Node(value=leaf_value)

        (
            X_left,
            y_left,
            X_right,
            y_right
        ) = self.split(
            X,
            y,
            feature,
            threshold
        )

        left_child = self.grow_tree(
            X_left,
            y_left,
            depth + 1
        )

        right_child = self.grow_tree(
            X_right,
            y_right,
            depth + 1
        )

        return Node(
            feature,
            threshold,
            left_child,
            right_child
        )


    def fit(self, X, y):

        self.root = self.grow_tree(X, y)


    def traverse_tree(
        self,
        x,
        node
    ):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:

            return self.traverse_tree(
                x,
                node.left
            )

        return self.traverse_tree(
            x,
            node.right
        )


    def predict(self, X):

        predictions = [
            self.traverse_tree(
                x,
                self.root
            )
            for x in X
        ]

        return np.array(predictions)