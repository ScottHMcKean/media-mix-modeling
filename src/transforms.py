"""
Transform functions for media mix modeling.

This module provides transformation functions for adstock effects and saturation curves
that can be used in both data generation and PyMC modeling.
"""

import pytensor.tensor as pt


def geometric_adstock(x, alpha, L=12, normalize=True):
    """
    Apply geometric adstock transformation to model carryover effects.

    Args:
        x: Input media spend vector
        alpha: Rate of decay (0-1)
        L: Length of carryover effect in time periods
        normalize: Whether to normalize weights

    Returns:
        Transformed spend vector with carryover effects
    """
    x = pt.as_tensor_variable(x).flatten()

    # Create convolution weights
    w = pt.power(alpha, pt.arange(L, dtype="float64"))

    if normalize:
        w = w / pt.sum(w)

    # Apply weighted sum of lags using a loop
    # Start with zeros
    result = pt.zeros(x.shape[0], dtype=x.dtype)

    for lag in range(L):
        # Create lagged version of x
        if lag == 0:
            lagged_x = x
        else:
            # Pad with zeros at the beginning and truncate at the end
            lagged_x = pt.concatenate([pt.zeros(lag, dtype=x.dtype), x[:-lag]])

        # Add weighted contribution
        result = result + w[lag] * lagged_x

    return result


def hill_saturation(x, k, s):
    """
    Apply Hill transformation to model diminishing returns.

    Args:
        x: Input media spend tensor
        k: Half-saturation constant
        s: Slope parameter (shape)

    Returns:
        Transformed values with saturation effect
    """
    x_tensor = pt.as_tensor_variable(x).flatten()
    numerator = pt.power(x_tensor, s)
    denominator = pt.power(k, s) + numerator
    return numerator / denominator


def logistic_saturation(x, mu):
    """
    Apply logistic saturation transformation.

    Args:
        x: Input media spend tensor
        mu: Rate of saturation

    Returns:
        Transformed spend vector
    """
    x_tensor = pt.as_tensor_variable(x).flatten()
    return (1 - pt.exp(-mu * x_tensor)) / (1 + pt.exp(-mu * x_tensor))
