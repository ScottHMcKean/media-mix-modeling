"""
Tests for transform functions.
"""

import numpy as np
import pytensor.tensor as pt

from src.transforms import geometric_adstock, logistic_saturation, hill_saturation


def test_geometric_adstock():
    """Test geometric adstock transformation."""
    n = 25
    a = 5
    x = np.zeros(n)
    x[a] = 1.0

    # Expected decay pattern with alpha=0.5
    y = np.zeros(n)
    y[a : a + 10] = np.array([0.5, 0.25, 0.125, 0.063, 0.031, 0.016, 0.008, 0.004, 0.002, 0.001])

    # Compute adstock
    y_hat = geometric_adstock(x, alpha=0.5, L=12, normalize=True).eval()

    assert np.allclose(y, y_hat, atol=0.01)


def test_logistic_saturation():
    """Test logistic saturation transformation."""
    n = 25
    x = np.linspace(0, 1, n)

    # Expected values with mu=0.5
    y = np.array(
        [
            0.000,
            0.010,
            0.021,
            0.031,
            0.042,
            0.052,
            0.062,
            0.073,
            0.083,
            0.093,
            0.104,
            0.114,
            0.124,
            0.135,
            0.145,
            0.155,
            0.165,
            0.175,
            0.185,
            0.195,
            0.205,
            0.215,
            0.225,
            0.235,
            0.245,
        ]
    )

    y_hat = logistic_saturation(x, mu=0.5).eval()

    assert np.allclose(y, y_hat, atol=0.01)


def test_hill_saturation():
    """Test Hill saturation transformation."""
    x = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])

    # Test with k=1.0, s=1.0
    y_hat = hill_saturation(x, k=1.0, s=1.0).eval()

    # At x=0, should be 0
    assert np.isclose(y_hat[0], 0.0, atol=0.01)

    # At x=k, should be 0.5
    assert np.isclose(y_hat[2], 0.5, atol=0.01)

    # Should be monotonically increasing
    assert np.all(np.diff(y_hat) >= 0)

    # Should approach 1 asymptotically
    assert y_hat[-1] < 1.0
    assert y_hat[-1] > 0.9


def test_geometric_adstock_no_normalize():
    """Test geometric adstock without normalization."""
    x = np.ones(20)

    y_norm = geometric_adstock(x, alpha=0.7, L=10, normalize=True).eval()
    y_no_norm = geometric_adstock(x, alpha=0.7, L=10, normalize=False).eval()

    # Without normalization should be larger
    assert np.all(y_no_norm >= y_norm)


def test_hill_saturation_shape_parameter():
    """Test Hill saturation with different shape parameters."""
    x = np.linspace(0, 5, 50)

    # Lower s means steeper curve
    y_steep = hill_saturation(x, k=1.0, s=0.5).eval()
    y_gradual = hill_saturation(x, k=1.0, s=2.0).eval()

    # Both should be between 0 and 1
    assert np.all(y_steep >= 0) and np.all(y_steep <= 1)
    assert np.all(y_gradual >= 0) and np.all(y_gradual <= 1)
