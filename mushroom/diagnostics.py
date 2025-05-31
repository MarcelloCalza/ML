"""
diagnostics.py

Helpers for model diagnostics: bias/variance aprrox. sweeps and plots.

### Functions:
- sweep_bias_variance: run a one-parameter sweep (with the other parameters being fixed), 
  measuring train, CV and test errors for each setting and saving the results to csv.
- plot_bias_variance: plot the saved errors and bias/variance estimated proxies.

Both assume binary classification with 0/1 labels and that ``build_model`` builds the 
model with the other parameters fixed.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def sweep_bias_variance(
        build_model,
        param_name,
        param_values,
        X, y, train_idx, test_idx,
        feature_types,
        cv_tune_fn,
        cv_k=10,
        csv_path="sweep_results.csv",
        seed=0):
    """
    Run a one-dimensional hyper-parameter sweep and gather error metrics.

    For each value ``v`` in param_values the function:

    - builds and fit a model via ``build_model`` on the training dataset;
    - computes 0-1 loss on on that same training dataset;
    - computes K-folds CV 0-1 loss (average over K folds) using a "single-point" grid 
       for the current value ``v`` (``build_model`` fixed the other hyperparams);
    - computes 0-1 loss on the test set for model built with current value ``v``;
    - saves the error metrics for all the inspected values of the varying parameter to
       a list that is later written to ``csv_path``.

    Args:
        build_model (Callable): callable that returns an un-fitted model with the fixed
            hyper-parameters.
        param_name (str): name of the hyper-parameter being swept.
        param_values (Sequence): iterable of values of the sweep parameter.
        X (np.ndarray): 2D samples array of shape
            (n (num of samples), d (num of features)).
        y (np.ndarray): 1D vector of the corresponding true labels.
        train_idx (np.ndarray): index for the traning split.
        test_idx (np.ndarray): index for the test split.
        feature_types (Sequence): passed through to ``cv_tune_fn`` so can be attached
            if the underlying learner needs the argument (tree).
        cv_tune_fn (Callable): K-fold CV function.
        cv_k (int): number of folds used by ``cv_tune_fn``. Defaults to 10.
        csv_path : filepath where the sweep metrics will be saved.
        seed : reproducibility seed forwarded to ``cv_tune_fn``. Defaults to 0.

    Returns:
        pandas.DataFrame: the table of error metrics for each value of the sweep param.
    """
    print("Starting model CV sweep for diagnostics")

    rows = []
    # Iterate through the sweep parameter possible values.
    for v in param_values:
        # Train the model on the training set using the current parameter value.
        model = build_model(**{param_name: v}).fit(X[train_idx], y[train_idx])
        
        # Compute 0-1 loss on the training set.
        train_err = np.mean(model.predict(X[train_idx]) != y[train_idx])

        # Create a "single-point" grid with the current value of the sweep parameter.
        grid = {param_name: [v]}

        # Generate K-fold CV average 0-1 loss for the model by running single level 
        # K-fold CV with the "single point" grid (not really CV anymore, it turns in more 
        # of a "K-fold validation" because the grid contains just 1 parameter)
        _, cv_err = cv_tune_fn(build_model, grid, X[train_idx], y[train_idx], feature_types=feature_types, K=cv_k, seed=seed)

        # Compute 0-1 loss on the test set.
        test_err = np.mean(model.predict(X[test_idx]) != y[test_idx])
        
        # Append 0-1 loss metrics to the output dict.
        rows.append({param_name: v, "train_error": train_err, "cv_error": cv_err, "test_error": test_err})
    
    # Turn output dict into a dataframe and save it.
    df = pd.DataFrame(rows).sort_values(param_name)
    df.to_csv(csv_path, index=False)
    print(f"Saved sweep metrics to {csv_path}")
    return df

def plot_bias_variance(df, param_name, title_prefix="", save_path=None):
    """
    Visualise the sweep results and simple bias/variance proxies.

    The function expects df to contain the columns produced by `sweep_bias_variance`.

    Two figures are produced:

    - raw errors: train, CV, test 0-1 loss against the swept parameter;
    - proxies: train loss (≈ bias) and CV loss - train loss (≈ variance).

    Args:
        df (pandas.DataFrame): dataFrame returned by `sweep_bias_variance`.
        param_name (str): name of the sweep hyperparameter.
        title_prefix (str): title for the figures.
        save_path (str): path where to save the plots.
    """
    df = df.copy()

    # Compute "variance" error proxy ≈ CV loss - training loss.
    df["variance_proxy"] = df["cv_error"] - df["train_error"]
    # "Bias" error proxy ≈ training loss.
    df["bias_proxy"] = df["train_error"]

    # Plot raw error curves against sweep parameter variation.
    plt.figure(figsize=(6,4))
    plt.plot(df[param_name], df["train_error"], marker="o", label="Train")
    plt.plot(df[param_name], df["cv_error"], marker="o", label="CV")
    plt.plot(df[param_name], df["test_error"], marker="o", label="Test")
    plt.xlabel(param_name)
    plt.ylabel("Error")
    plt.title(f"{title_prefix} Train / CV / Test error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}_raw_errors.png", dpi=300)
    plt.show()
    
    # Plot bias and variance curves against sweep parameter variation.
    plt.figure(figsize=(6,4))
    plt.plot(df[param_name], df["bias_proxy"], marker="o", label="Bias proxy (train)")
    plt.plot(df[param_name], df["variance_proxy"], marker="o", label="Variance proxy (cv-train)")
    plt.xlabel(param_name)
    plt.ylabel("Error proxy")
    plt.title(f"{title_prefix} Bias-Variance trade-off")
    plt.legend() 
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}_bias_variance_approx.png", dpi=300)
    plt.show()