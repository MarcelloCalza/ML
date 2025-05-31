"""
run.py

Main scripts run for the project.

The pipeline:

-  Download the UCI mushroom classification data set (through ``ucimlrepo`` helper).
-  Clean + encode the raw data into a samples matrix ``X`` and a true labels vector ``y``
   (0 = edible, 1 = poisonous).
-  Tune a ``DecisionTree`` with 10-fold CV to find the best hyper-parameters in the grid.
-  Retrain the tuned tree on the full training split and save train/CV/test 0-1 losses.
-  Tune a ``RandomForest`` with 10-fold CV (keeping the best tree hyper-params fixed)
   to find the best RF hyper-parameters in the grid.
-  Retrain the tuned RF on the full training split and save train/CV/test 0-1 losses.
-  Produce varying 0-1 losses metrics over a sweep parameter (depth for the tree, number
   of estimators for the forest) and generate corresponding bias and variance approx plots.

All intermediate results are stored in the ``data/`` directory, and the plots are stored in
the ``plots/`` directory.
"""

from mushroom.tree import DecisionTree
from mushroom.forest import RandomForest
from mushroom.tuning import cv_tune
from mushroom.diagnostics import sweep_bias_variance, plot_bias_variance
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import os


if __name__ == "__main__":

    # Create folder to save data.
    os.makedirs("data", exist_ok=True)

    # Create folder to save plots.
    os.makedirs("plots", exist_ok=True)

    # Fix RNG seed for reproducibility.
    seed = 31052025

    # Retrieve the UCI mushroom dataset.
    ds = fetch_ucirepo(id=848)

    # Extract features dataset and targets as pandas objects.
    X_df = ds.data.features.copy()
    y_ser = ds.data.targets.squeeze()

    # Handle eventual missing values by filling a special category.
    missing_token = ds.metadata.missing_values_symbol
    X_df.replace(missing_token, np.nan, inplace=True)
    X_df = X_df.fillna("MISSING")

    # Retrieve feature types and build feature_types list (needed by the DecisionTree).
    varinfo = ds.variables.set_index("name")["type"].to_dict()
    feature_types = [
        (
            "numeric"
            if varinfo[col].lower() in ("continuous", "integer", "real", "ordinal")
            else "categorical"
        )
        for col in X_df.columns
    ]

    # Convert categorical columns to integer codes for modeling.
    X = X_df.apply(lambda c: c.astype("category").cat.codes).to_numpy()

    # Convert targets into integer labels (0 = edible, 1 = poisonous).
    y = pd.Categorical(y_ser).codes

    # Shuffle and split indices into 80% train and 20% test.
    idx = np.random.default_rng(seed).permutation(len(y))
    train_idx, test_idx = idx[: int(0.8 * len(y))], idx[int(0.8 * len(y)) :]

    # Set up grid of tree hyper-parameters to search over.
    grid = {
        "criterion": ["psi2", "psi3", "psi4"],
        "max_depth": [15, 20, 25],
        "min_samples_leaf": [10, 15, 20],
        "min_impurity_decrease": [1e-2, 1e-4, 1e-6],
    }
    
    # Tune a decision tree using 10 folds single level Cross Validation.
    best_params, best_cv_err = cv_tune(
        DecisionTree,
        grid,
        X[train_idx],
        y[train_idx],
        feature_types=feature_types,
        K=10,
        seed=seed,
    )
    print("Best params via CV:", best_params, "CV error:", best_cv_err)

    # Retrain tuned tree on all training data.
    final_tree = DecisionTree(**best_params).fit(X[train_idx], y[train_idx])
    # Compute and save tree 0-1 loss on training set.
    train_err = np.mean(final_tree.predict(X[train_idx]) != y[train_idx])
    print("Final tree train error:", train_err)
    # Compute and save tree 0-1 loss on test set.
    test_err = np.mean(final_tree.predict(X[test_idx]) != y[test_idx])
    print("Final tree test error:", test_err)

    # Remove "feature_types" if present.
    params_to_save = {k: v for k, v in best_params.items() if k != "feature_types"}

    # Generate tree results dataframe.
    results = pd.DataFrame(
        [
            {
                **params_to_save,
                "train_error": train_err,
                "cv_error": best_cv_err,
                "test_error": test_err,
            }
        ]
    )

    # Save results to CSV.
    results.to_csv("data/tree_run_results.csv", index=False)
    print(
        "Saved best tree paremeters and best tree error metrics to tree_run_results.csv"
    )
    # Retrieve tree run results.
    best_tree_df = pd.read_csv("data/tree_run_results.csv")

    # Fix all other hyper-params at their "best" values.
    best_tree_params = {
        "criterion": best_tree_df["criterion"].iloc[0],
        "max_depth": best_tree_df["max_depth"].iloc[0],
        "min_samples_leaf": best_tree_df["min_samples_leaf"].iloc[0],
        "min_impurity_decrease": best_tree_df["min_impurity_decrease"].iloc[0],
    }

    # Tree builder that injects the fixed best params except max_depth (max tree depth).
    def build_tree(max_depth, **_):
        return DecisionTree(
            max_depth=max_depth,
            criterion=best_tree_df["criterion"].iloc[0],
            min_samples_leaf=best_tree_df["min_samples_leaf"].iloc[0],
            min_impurity_decrease=best_tree_df["min_impurity_decrease"].iloc[0],
            feature_types=feature_types,
        )

    # Define max tree depths for the tree sweep.
    depths = [1, 5, 10, 15, 20, 25, 30]

    # Compute tree max depth sweep and retrieve losses metrics.
    tree_df = sweep_bias_variance(
        build_tree,
        "max_depth",
        depths,
        X,
        y,
        train_idx,
        test_idx,
        feature_types=feature_types,
        cv_tune_fn=cv_tune,
        cv_k=10,
        csv_path="data/tree_sweep.csv",
        seed=seed,
    )

    tree_sweep = pd.read_csv("data/tree_sweep.csv")

    # Tree bias / variance approximations diagnostic plots over varying max tree depths.
    plot_bias_variance(
        tree_sweep,
        "max_depth",
        title_prefix="Decision Tree:",
        save_path="plots/tree_sweep",
    )

    # Set up grid of random forest hyper-parameters to search over.
    rf_grid = {
        "n_estimators": [10, 25, 50],
        "max_features": [int(np.sqrt(X.shape[1])), int(X.shape[1] / 3)],
    }

    # Helper that builds a forest with the best tree settings
    # (needed to be able to reuse cv_tune for random forests).
    def build_rf(n_estimators: int, max_features: int, **_) -> RandomForest:
        return RandomForest(
            n_estimators=n_estimators,
            max_features=max_features,
            tree_kwargs=best_tree_params,
            seed=seed,
        )

    # Tune a random forest using 10 folds single level Cross Validation.
    best_rf_params, best_rf_cv = cv_tune(
        build_rf,
        rf_grid,
        X[train_idx],
        y[train_idx],
        feature_types=feature_types,
        K=10,
        seed=seed,
    )
    print("Best RF params:", best_rf_params, "Best RF CV error:", best_rf_cv)
    
    # Retrain tuned random forest on all training data.
    final_rf = build_rf(**best_rf_params).fit(X[train_idx], y[train_idx])

    # Compute and save random forest 0-1 loss on training set.
    rf_train_err = np.mean(final_rf.predict(X[train_idx]) != y[train_idx])
    print("Final RF train error:", rf_train_err)
    # Compute and save random forest 0-1 loss on test set.
    rf_test_err = np.mean(final_rf.predict(X[test_idx]) != y[test_idx])
    print("Final RF test error:", rf_test_err)

    # Generate random forest results dataframe.
    rf_results = pd.DataFrame(
        [
            {
                **best_tree_params,
                **best_rf_params,
                "train_error": rf_train_err,
                "cv_error": best_rf_cv,
                "test_error": rf_test_err,
            }
        ]
    )

    # Save results to CSV.
    rf_results.to_csv("data/rf_run_results.csv", index=False)
    print(
        "Saved best random forest paremeters and best random forest error metrics to rf_run_results.csv"
    )
    
    # Retrieve tuned random forest run metrics.
    best_rf_df = pd.read_csv("data/rf_run_results.csv")

    # Random forest builder that injects the fixed best params except n_estimators (number of trees for rf).
    def build_forest(n_estimators, **kwargs):
        return RandomForest(
            n_estimators=n_estimators,
            max_features=best_rf_df["max_features"].iloc[0],
            tree_kwargs=dict(
                criterion=best_rf_df["criterion"].iloc[0],
                max_depth=best_rf_df["max_depth"].iloc[0],
                min_samples_leaf=best_rf_df["min_samples_leaf"].iloc[0],
                min_impurity_decrease=best_rf_df["min_impurity_decrease"].iloc[0],
                feature_types=feature_types,
            ),
            seed=seed,
        )

    # Define number of trees for the random forest sweep.
    sizes = [1, 5, 10, 25, 50, 75, 100]

    # Compute random forest number of trees sweep and retrieve losses metrics.
    rf_df = sweep_bias_variance(
        build_forest,
        "n_estimators",
        sizes,
        X,
        y,
        train_idx,
        test_idx,
        feature_types=feature_types,
        cv_tune_fn=cv_tune,
        cv_k=10,
        csv_path="data/rf_sweep.csv",
        seed=seed,
    )
    
    rf_sweep = pd.read_csv("data/rf_sweep.csv")

    # Tree bias / variance approximations diagnostic plots over varying n_estimators (number of trees).
    plot_bias_variance(
        rf_sweep,
        "n_estimators",
        title_prefix="Random Forest:",
        save_path="plots/rf_sweep",
    )
