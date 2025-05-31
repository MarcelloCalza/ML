"""
tree.py

Core classes and functions for binary decision-tree classifiers.

### Classes:
- Node: a light container representing one node (either an internal split or a leaf)
  in the tree.
- DecisionTree: greedy learner that supports single-feature binary tests (threshold on
  numerics / equality on categoricals) and several stopping criteria.

A tree is composed of Node objects linked through left / right attributes. Each node
stores either:

- split logic: a callable g(X) -> bool that tells whether every row in X should go left
  (True) or right (False), plus metadata describing the feature and value used in the test;
  or:
- leaf information: a predicted class label.

It supports both categorical and numeric features, a selection of impurity criteria
(see criteria.py), early stopping criteria such as max_depth (max tree depth),
min_samples_leaf (min samples per leaf) and min_impurity_decrease, and the optional
max_features (for Random-Forest) to restrict the number of candidate columns examined at
each split.
"""

import numpy as np
from collections import Counter
from typing import Optional, Callable, Union
from .criteria import CRITERIA


class Node:
    """
    A single node in the decision tree. The node can be internal holding a
    splitting rule or terminal (leaf) holding a prediction.

    Args:
        g (Callable[[np.ndarray], np.ndarray]):
            Vectorised function that, given a batch of samples ``X`` of shape (n, d),
            returns a boolean mask selecting rows that should go to the left child.
            ``None`` for leaf nodes.
        feature (int): index of the feature used in the split.
        value (int or float): threshold or category defining the splitting rule.
        is_leaf (bool): if ``True`` the node is a leaf.
        prediction (int): class label stored in a leaf.
    """

    def __init__(
        self,
        g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        feature: Optional[int] = None,
        value: Optional[Union[int, float]] = None,
        *,
        is_leaf: bool = False,
        prediction: Optional[int] = None,
    ):
        self.g = g
        self.feature = feature
        self.value = value
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.is_leaf = is_leaf
        self.prediction = prediction

    def __call__(self, X: np.ndarray) -> bool:
        """
        Apply the node's decision function to a batch of samples.

        Args:
            X (np.ndarray): 2D samples array of shape
                (n (num of samples), d (num of features)).

        Returns:
            np.ndarray: Boolean mask of length n, ``False`` if the node is a leaf
                (no decision to make).
        """
        if self.g:
            # Apply split function to all samples in x.
            return self.g(X)
        # Otherwise, default to no split (treat as false mask).
        return np.zeros(len(X), dtype=bool)


class DecisionTree:
    """
    A greedy, binary decision-tree classifier.

    The tree is grown top-down by maximising the reduction in the chosen
    impurity criterion at each node.

    Args:
        criterion (str): key selecting the impurity measure.
        max_depth (int): maximum tree depth allowed.
        min_samples_leaf (int): minimum number of training samples that must remain in
            each child after a split.
        min_impurity_decrease (float): do not split if the impurity gain is
            below this threshold.
        feature_types (list[str]): List of "numeric" or "categorical" strings telling
            how every column should be treated. If ``None`` everything is assumed
            categorical.
        max_features (int): size of the subset of features used to search splits for
            Random Forests.
    """

    def __init__(
        self,
        criterion: str = "psi2",
        max_depth: int = None,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        feature_types: Optional[list[str]] = None,
        max_features=None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.feature_types = feature_types
        self.root: Optional[Node] = None
        self._impurity = CRITERIA[criterion]
        self.max_features = max_features

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Grow a tree that classifies X -> y.

        Args:
            X (np.ndarray): 2D design matrix (n, d).
            y (np.ndarray): 1D vector of binary labels.

        Returns:
            DecisionTree: the fitted instance (for chaining).
        """
        # Record number of classes.
        self.n_classes_ = len(np.unique(y))

        # Cache the global unique values of each column.
        if self.feature_types:
            self._cat_uniques = [
                np.unique(X[:, j]) if self.feature_types[j] == "categorical" else None
                for j in range(X.shape[1])
            ]
        else:
            self._cat_uniques = None

        # Build the tree recursively starting at depth 0.
        self.root = self._grow(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Label each row in X by traversing the tree.

        Args:
            X (np.ndarray): 2D array of samples.

        Returns:
            np.ndarray: 1D array of predicted labels.
        """
        # Call the helper _infer on each sample to follow splits.
        return np.asarray([self._infer(x, self.root) for x in X])

    def _grow(self, X, y, depth) -> Node:
        """
        Recursively build (or expand) a subtree.

        Args:
            X (np.ndarray): 2D array of training samples routed to the current node.
            y (np.ndarray): corresponding array of true labels.
            depth (int): depth of the current node (root = 0).

        Returns:
            Node: the root node of the frbuilt subtree (can be a leaf if no split was 
            possible).
        """
        n_samples = len(y)

        # Current node impurity before splitting.
        current_imp = self._impurity(y)

        # Stop splitting if any stopping condition is met.
        if (
            # Reached max tree depth.
            (self.max_depth is not None and depth >= self.max_depth)
            # Dont have enough samples for each leaf (complete binary tree).
            or n_samples < 2 * self.min_samples_leaf
            # Current impurity is 0 (all labels in y are the same)
            or current_imp == 0.0
        ):
            # Make a leaf with the majority class of y.
            return Node(is_leaf=True, prediction=self._majority(y))

        if self.max_features is None:
            feature_candidates = range(X.shape[1])  # use *all* columns
        else:
            rng = np.random.default_rng()  # fresh RNG each call
            feature_candidates = rng.choice(
                X.shape[1], size=min(self.max_features, X.shape[1]), replace=False
            )

        # Variables to track the current best split.
        best_gain, best_feat, best_val, best_mask = 0, None, None, None

        def _update_best(mask, j, val):
            """
            Helper function that evaluates a candidate split and if its the current best
            remembers it.

            Args:
                mask (np.ndarray): boolean array marking samples that would go left with
                    ``True`` given the candidate split.
                j (int): column index (feature idx) the split is applied to.
                val (float or int): threshold (for numeric) or category value
                    (for categorical) that defines the split.
            """
            nonlocal best_gain, best_feat, best_val, best_mask
            # Enforce minimum samples in each child.
            if (
                mask.sum() < self.min_samples_leaf
                or (~mask).sum() < self.min_samples_leaf
            ):
                return
            # Compute impurities of the two resulting subsets.
            imp_left = self._impurity(y[mask])
            imp_right = self._impurity(y[~mask])
            # Weighted impurity after split.
            weighted_imp = (mask.sum() / n_samples) * imp_left + (
                (~mask).sum() / n_samples
            ) * imp_right
            # Compute gain.
            gain = current_imp - weighted_imp
            # Check if the current gain is the best.
            if gain > best_gain:
                best_gain, best_feat, best_val, best_mask = gain, j, val, mask

        # Try every feature and every possible binary test.
        for j in feature_candidates:
            X_col = X[:, j]
            # Determine split type for feature j.
            ftype = self.feature_types[j] if self.feature_types else "categorical"
            # Check feature type.
            if ftype == "categorical":
                # Use cached uniques (possibly).
                if self._cat_uniques is not None:
                    values = self._cat_uniques[j]
                else:
                    values = np.unique(X_col)
                # Iterate through every possible value of j.
                for v in values:
                    # Boolean mask thats true for every sample which has j = v.
                    mask = X_col == v
                    # Update best gain (possibly).
                    _update_best(mask, j, v)
            else:
                # Retrieve every unique value for feature j.
                vals = np.unique(X_col)
                # If there is only one unique value continue.
                if len(vals) <= 1:
                    continue
                # Form thresholds to separate each adjacent values.
                thresholds = (vals[:-1] + vals[1:]) / 2.0
                # Iterate through every threshold.
                for t in thresholds:
                    # Boolean mask thats true for every sample which has j <= t.
                    mask = X_col <= t
                    # Update best gain (possibly).
                    _update_best(mask, j, t)

        # Check if best gain satisfy the minimum impurity criteria.
        if best_gain >= self.min_impurity_decrease:
            # Build the correct split function based on feature type.
            if self.feature_types and self.feature_types[best_feat] == "numeric":
                split_fun = lambda x, j=best_feat, t=best_val: x[:, j] <= t
            else:
                split_fun = lambda x, j=best_feat, v=best_val: x[:, j] == v
            # Create the node and recurse on its children.
            node = Node(g=split_fun, feature=best_feat, value=best_val, is_leaf=False)
            node.left = self._grow(X[best_mask], y[best_mask], depth + 1)
            node.right = self._grow(X[~best_mask], y[~best_mask], depth + 1)
            return node

        # If no valid split found create leaf.
        return Node(is_leaf=True, prediction=self._majority(y))

    def _infer(self, x, node):
        """
        Predict the label for one sample by tree traversal.

        Args:
            x (np.ndarray): 1D feature vector of the sample.
            node (Node): node from which to start.

        Returns:
            int: predicted class label (0 or 1) stored in the reached leaf.
        """
        # Traverse the tree until a leaf is reached.
        while not node.is_leaf:
            # Retrieve current node one dimensional boolean mask for the single sample x.
            go_left = node(x[np.newaxis, :])[0]
            # Pick the branch to traverse based on the boolean mask.
            node = node.left if go_left else node.right

        return node.prediction

    @staticmethod
    def _majority(y):
        """
        Return the majority class of a true label vector.

        Args:
            y (np.ndarray): 1D array of true binary labels.

        Returns:
            int: 0 or 1, whichever appears more often in ``y``.
        """
        return int(Counter(y).most_common(1)[0][0])
