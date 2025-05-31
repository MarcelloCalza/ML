"""
criteria.py

Utility functions that implement three impurity (split quality) criteria.

Each function takes the label vector of the data that reach a node and returns an 
impurity value, non-negative number that is zero when the node is pure and grows as the 
class mix becomes more heterogeneous.

### Functions:
- pos_fraction: fraction of positive labels
- psi2: Gini impurity index.
- psi3: information gain (scaled entropy).
- psi4: square root impurity.
All functions assume labels are encoded as 0 (negative) and 1 (positive).
"""

import numpy as np


def pos_fraction(y: np.ndarray) -> float:
    """
    Compute the fraction p of positive labels in y.

    Args:
        y (np.ndarray): 1D array of binary labels.

    Returns:
        float: the fraction ``p = (# of ones) / len(y)``.
    """
    # Count how many labels are positive.
    N_pos = np.count_nonzero(y == 1)
    # Total number of labels.
    N_tot = y.size

    return N_pos / N_tot


def psi2(y: np.ndarray) -> float:
    """
    psi2: Gini index = 2*p*(1-p).
    Where ``p`` is the output of `pos_fraction`.

    Args:
        y (np.ndarray): 1D array of binary labels.

    Returns:
        float: Gini impurity index. 
    """
    # Compute fraction of positives.
    p = pos_fraction(y)

    return 2.0 * p * (1.0 - p)


def psi3(y: np.ndarray) -> float:
    """
    psi3: scaled information gain = -1/2[p log2 p + (1-p) log2(1-p)].
    Where ``p`` is the output of `pos_fraction`.

    Args:
        y (np.ndarray): 1D array of binary labels.

    Returns:
        float: scaled entropy impurity. Zero for a pure node.
    """
    # Compute fraction of positives.
    p = pos_fraction(y)
    if p in (0.0, 1.0):
        # If all labels are identical, impurity is zero.
        return 0.0

    return -0.5 * (p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def psi4(y: np.ndarray) -> float:
    """
    psi4: square root impurity = sqrt(p*(1-p)).
    Where ``p`` is the output of `pos_fraction`.

    Args:
        y (np.ndarray): 1D array of binary labels.
    
     Returns:
        float: square root impurity.
    """
    # Compute fraction of positives.
    p = pos_fraction(y)

    return np.sqrt(p * (1.0 - p))

# Registry for lookup.
CRITERIA = {"psi2": psi2, "psi3": psi3, "psi4": psi4}
