from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _validate_option_type(option_type: str) -> None:
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")


def simulate_terminal_prices(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    simulations: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate terminal stock prices under the risk-neutral measure."""
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if simulations <= 0:
        raise ValueError("simulations must be positive.")

    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 1.0, simulations)
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * shocks)


def monte_carlo_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    simulations: int,
    option_type: OptionType = "call",
    seed: int | None = None,
) -> float:
    """Price a European option by discounting average simulated payoff."""
    _validate_option_type(option_type)
    terminal_prices = simulate_terminal_prices(S0, r, sigma, T, simulations, seed)

