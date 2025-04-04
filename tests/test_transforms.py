import numpy as np
from mediamix.transforms import geometric_adstock, logistic_saturation


def test_geometric_adstock():
    n = 25
    a = 5
    x = np.zeros(n)
    x[a] = 1.0
    y = np.zeros(n)
    y[a : a + 10] = np.array(
        [0.5, 0.25, 0.125, 0.063, 0.031, 0.016, 0.008, 0.004, 0.002, 0.001]
    )
    y_hat = geometric_adstock(x, 0.5).eval()
    assert np.allclose(y, y_hat, atol=0.01)


def test_logisticsaturation():
    n = 25
    x = np.linspace(0, 1, n)
    y = [
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
    y_hat = logistic_saturation(x, 0.5).eval()
    assert np.allclose(y, y_hat, atol=0.01)
