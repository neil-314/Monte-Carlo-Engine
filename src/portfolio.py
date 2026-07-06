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
    if len(expected_returns) != asset_count or len(volatilities) != asset_count:
        raise ValueError("weights, expected_returns, and volatilities must have equal length.")
    if correlation_matrix.shape != (asset_count, asset_count):
        raise ValueError("correlation_matrix must be square with one row/column per asset.")
    if np.any(volatilities < 0):
        raise ValueError("volatilities cannot be negative.")
    if not np.allclose(correlation_matrix, correlation_matrix.T):
        raise ValueError("correlation_matrix must be symmetric.")
    if not np.allclose(np.diag(correlation_matrix), 1.0):
        raise ValueError("correlation_matrix must have ones on the diagonal.")


def simulate_correlated_portfolio(
    initial_value: float,
    weights: np.ndarray,
    expected_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    T: float = 1.0,
    simulations: int = 100_000,
    seed: int | None = None,
    asset_names: list[str] | None = None,
) -> pd.DataFrame:
    """Simulate terminal portfolio values using correlated lognormal asset returns."""
    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if simulations <= 0:
        raise ValueError("simulations must be positive.")

    weights = np.asarray(weights, dtype=float)
    expected_returns = np.asarray(expected_returns, dtype=float)
    volatilities = np.asarray(volatilities, dtype=float)
    correlation_matrix = np.asarray(correlation_matrix, dtype=float)
    validate_portfolio_inputs(weights, expected_returns, volatilities, correlation_matrix)

    if asset_names is None:
        asset_names = [f"Asset {index + 1}" for index in range(len(weights))]

    rng = np.random.default_rng(seed)
    independent_shocks = rng.normal(size=(simulations, len(weights)))
    cholesky_matrix = np.linalg.cholesky(correlation_matrix)
    correlated_shocks = independent_shocks @ cholesky_matrix.T

    log_returns = (
        (expected_returns - 0.5 * volatilities**2) * T
        + volatilities * np.sqrt(T) * correlated_shocks
    )
    asset_returns = np.exp(log_returns) - 1
    portfolio_returns = asset_returns @ weights
    portfolio_values = initial_value * (1 + portfolio_returns)

    data = {
        f"{name} Return": asset_returns[:, index]
        for index, name in enumerate(asset_names)
    }
    data["Portfolio Return"] = portfolio_returns
    data["Portfolio Value"] = portfolio_values
    return pd.DataFrame(data)


def plot_portfolio_distribution(
    portfolio_returns: np.ndarray,
    output_path: str | Path,
    confidence_level: float = 0.95,
) -> None:
    """Plot portfolio returns with expected return, VaR, and CVaR markers."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_return = np.mean(portfolio_returns)
    var_return = calculate_var(portfolio_returns, confidence_level)
    cvar_return = calculate_cvar(portfolio_returns, confidence_level)

    plt.figure(figsize=(11, 6))
    plt.hist(
        portfolio_returns * 100,
        bins=70,
        color="#72B7B2",
        alpha=0.85,
        edgecolor="white",
    )
    plt.axvline(
        expected_return * 100,
        color="#F58518",
        linewidth=2,
        label=f"Expected return: {expected_return:.2%}",
    )
    plt.axvline(
        var_return * 100,
        color="#E45756",
        linestyle="--",
        linewidth=2,
        label=f"{confidence_level:.0%} VaR: {var_return:.2%}",
    )
    plt.axvline(
        cvar_return * 100,
        color="#B279A2",
        linestyle=":",
        linewidth=2.5,
        label=f"{confidence_level:.0%} CVaR: {cvar_return:.2%}",
    )
    plt.title("Portfolio Return Distribution")
    plt.xlabel("Portfolio Return (%)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.22)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


