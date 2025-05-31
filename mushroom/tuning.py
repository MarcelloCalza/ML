"""
tuning.py

Generic module for cross-validation and hyper-parameters search.

The module is deliberately model-agnostic because it is used for both trees and
random forests: it never inspects the internals of a learner.  All it needs is a factory
that, given a dictionary of keyword arguments, returns an unfitted model.

### Functions:
- k_fold_indices: split an index range into K shuffled folds.
- cv_tune: grid search with single-level K-fold Cross Validation.
"""

import numpy as np
import gc, time, inspect
from tqdm.auto import tqdm
from itertools import product


def k_fold_indices(n, K, seed=0):
    """
    Return a list of K mutually exclusive index arrays.

    Args:
        n (int): total number of samples.
        K (int): number of folds.
        seed (int): RNG seed for shuffling. Defaults to ``0``.

    Returns:
        list[np.ndarray]: a K length list where each element is a 1D array of indices
            belonging to that fold. The length of the union of the arrays equals to ``n``
            and the intersection is empty.
    """
    # Set RNG seed for reproducibility.
    rng = np.random.default_rng(seed)
    # Shuffle the samples.
    idx = rng.permutation(n)
    return np.array_split(idx, K)


def cv_tune(model_factory, grid, X_train, y_train, *, feature_types, K=10, seed=0):
    """
    Parameters grid search with single-level K-fold Cross Validation.

    The function tries every hyper-parameter combo in ``grid``. For each combo it
    performs single level K-fold CV and records the average 0-1 classification error.
    The combo with the smallest CV error is returned.

    Args:
        model_factory (Callable): a callable such that ``model_factory(**params)``
            returns a new unfitted model.
        grid (dict[str, Sequence]): dictionary that maps hyper-parameters names ->
            sequence of possible values.
        X_train (np.ndarray): training samples matrix of shape
            ``(n (number of samples), d (number of features))``.
        y_train (np.ndarray): corresponding 1D vector of true binary labels.
        feature_types (Sequence[str]): optional list needed only if the model callable
            needs a ``feature_types`` parameter (for the tree).
        K (int): number of CV folds. Defaults to 10.
        seed (int, optional): seed used for fold shuffling. Defaults to 0.

    Returns:
        tuple: ``(best_params, best_cv_error)`` where:
            - ``best_params`` : the winning hyper-parameters combination.
            - ``best_cv_error`` : the mean 0-1 loss across the K folds.
    """
    best_params, best_err = None, 1.0
    # Generate the K folds indices.
    folds = k_fold_indices(len(y_train), K, seed=seed)
    # Compute cartesian products of the hyper-params grid to generate every combination.
    combos = list(product(*grid.values()))

    # Iterate through the hyper-parameters combinations.
    for combo in tqdm(
        combos,
        total=len(combos),
        desc=f"Single Level {K}-fold CV",
        unit="combo",
        mininterval=1.0,
        position=0,
    ):
        t0 = time.perf_counter()

        # Turn params combo into argument dictionary for the model.
        params = dict(zip(grid.keys(), combo))

        # Only attach feature_types if the model can accept it.
        if feature_types is not None:
            sig = inspect.signature(model_factory)
            if "feature_types" in sig.parameters:
                params["feature_types"] = feature_types

        errs = []
        # Iterate through the K folds.
        for i, valid_idx in enumerate(folds):
            # Generate training indices = all except fold i (Strain = S \ Si).
            train_fold = np.hstack([f for j, f in enumerate(folds) if j != i])
            # Fit model on Strain = S \ Si.
            model = model_factory(**params).fit(
                X_train[train_fold], y_train[train_fold]
            )
            # Measure 0-1 loss on validation fold Si.
            errs.append(
                np.mean(model.predict(X_train[valid_idx]) != y_train[valid_idx])
            )
        # Compute average 0-1 across the folds for the current hyper-parameters combo.
        avg_err = sum(errs) / K
        # Update best hyper-parameters combo and average 0-1 loss.
        if avg_err < best_err:
            best_params, best_err = params, avg_err

        # Timing info for the current hyper-param combo.
        took = time.perf_counter() - t0
        tqdm.write(f"combo={combo}   err={avg_err:.4f}   {took:.2f}s")
        # Free memory.
        gc.collect()

    return best_params, best_err
