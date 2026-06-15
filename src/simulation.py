from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    simulations: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate stock-price paths using Geometric Brownian Motion."""
    if S0 <= 0:
