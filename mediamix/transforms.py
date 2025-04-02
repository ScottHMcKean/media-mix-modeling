import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as tt
import warnings

warnings.filterwarnings("ignore", "iteritems is deprecated")
warnings.filterwarnings("ignore", "`np.complex` is a deprecated")


def geometric_adstock(x, alpha=0, L=12, normalize=True):
    """Geometric Adstock Function.

    :param x: Input media spend vector
    :param alpha: Rate of decay (float)
    :param L: Length of time carryover effects can have an impact (int)
    :param normalize: Whether to normalize the weights (bool)
    :return: Transformed spend vector
    """

    w = tt.as_tensor_variable([tt.power(alpha, i) for i in range(L)])
    xx = tt.stack(
        [tt.concatenate([tt.zeros(i), x[: x.shape[0] - i]]) for i in range(L)]
    )

    if not normalize:
        y = tt.dot(w, xx)
    else:
        y = tt.dot(w / tt.sum(w), xx)
    return y


def sigmoid_saturation(x_t, mu=0.1):
    """
    Theano implementation of sigmoid saturation function.

    :param x_t: Input media spend tensor
    :param mu: Rate of saturation (float)
    :return: Transformed spend vector
    """
    return 1 / (1 + tt.exp(-mu * x_t))


def logistic_saturation(x_t, mu=0.1):
    """
    Theano implementation of nonlinear saturation function.

    :param x_t: Input media spend tensor
    :param mu: Rate of saturation (float)
    :return: Transformed spend vector
    """
    return (1 - tt.exp(-mu * x_t)) / (1 + tt.exp(-mu * x_t))


def hill_saturation(x_t, k=0.5, s=0.7):
    """Apply Hill transformation to model diminishing returns using PyTensor.

    :param x_t: Input media spend tensor
    :param k: Half-saturation constant
    :param s: Slope parameter
    :return: Transformed values with saturation effect
    """
    x = tt.as_tensor_variable(x_t)

    mask = x > 0

    result = tt.zeros_like(x)
    result = tt.set_subtensor(result[mask], x[mask] ** s / (k**s + x[mask] ** s))

    return result
