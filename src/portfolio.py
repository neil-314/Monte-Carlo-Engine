from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk_metrics import calculate_cvar, calculate_var, portfolio_risk_report


def format_dollars(value: float, decimals: int = 0) -> str:
    """Format dollar amounts with the minus sign before the currency symbol."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def validate_portfolio_inputs(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
) -> None:
    """Validate portfolio array shapes and common numerical constraints."""
    asset_count = len(weights)
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")
