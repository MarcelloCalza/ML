"""
forest.py

Random forest learner that wraps the ``DecisionTree`` class in a Random-Forest scheme.

### Classes:
- RandomForest: builds n_estimators decision trees, each trained on a random subsample 
  from the dataset S with replacements (same size n of S but drawn with replacements 
  using a uniform distribution over S) and, at every split, restricted to a random 
  subset of features (max_features). Predictions are obtained by majority vote.

Only binary classification (labels 0/1) is supported because the base learner is binary.
"""

import numpy as np
from tqdm.auto import tqdm
from .tree import DecisionTree
import gc

class RandomForest:
    """
    Random-Forest classifier built on top of the DecisionTree implementation.
    Args:
        n_estimators (int): number of trees to build (default 10).
        max_features (int): size k of the random feature subset examined per split 
            inside each tree.
        tree_kwargs (dict): extra keyword arguments forwarded to every underlying
            DecisionTree class.
        trees (list[DecisionTree]): the fitted base tree estimators.
        seed (int): RNG seed for dataset subsampling. Defaults to 0.
    """
    def __init__(self, n_estimators=10, max_features=None, *, tree_kwargs=None, seed=0):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.tree_kwargs = tree_kwargs or {}
        self.seed = seed
        self.trees: list[DecisionTree] = []

    def fit(self, X, y):
        """
        Train n_estimators independent trees.

        Args:
            X (np.ndarray): 2D samples array of shape
                (n (num of samples), d (num of features)).
            y (np.ndarray): corresponding 1D array of true labels.

        Returns:
            RandomForest: the fitted instance (enables chaining).
        """
        n = len(y)
        rng = np.random.default_rng(self.seed)
        self.trees = []

        bar = tqdm(range(self.n_estimators), desc="building trees for RF", leave=False, unit="tree", position=1)
        
        # Iterate through the number trees needed for the rf (n_estimators).
        for _ in bar:

            # Sample with replacement from the dataset.
            idx = rng.integers(0, n, n)
            X_boot, y_boot = X[idx], y[idx]

            # Build and fit one tree.
            tree = DecisionTree(max_features=self.max_features, **self.tree_kwargs)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            # Free memory.
            gc.collect()

        return self

    def predict(self, X):
        """
        Majority-vote predictions for a batch of samples.

        Args:
            X (np.ndarray): 2D array samples to classify with shape (n, d).

        Returns:
            np.ndarray: 1D array of predicted labels.
        """
        # Collect predictions from every tree -> shape (n_samples, n_estimators).
        preds = np.column_stack([t.predict(X) for t in self.trees])
        # Majority vote (ties broken at 0).
        return (preds.mean(axis=1) >= 0.5).astype(int)