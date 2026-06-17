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
        raise ValueError("S0 must be positive.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if simulations <= 0:
        raise ValueError("simulations must be positive.")

    rng = np.random.default_rng(seed)
    dt = T / steps
    shocks = rng.normal(0.0, 1.0, size=(steps, simulations))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks

    paths = np.empty((steps + 1, simulations))
    paths[0] = S0
    paths[1:] = S0 * np.exp(np.cumsum(log_returns, axis=0))
    return paths


def terminal_price_statistics(
    terminal_prices: np.ndarray,
    starting_price: float,
    target_price: float,
) -> dict[str, float]:
    """Calculate summary statistics from simulated terminal prices."""
    terminal_prices = np.asarray(terminal_prices)
    return {
        "starting_price": float(starting_price),
        "mean_terminal_price": float(np.mean(terminal_prices)),
        "median_terminal_price": float(np.median(terminal_prices)),
        "terminal_std_dev": float(np.std(terminal_prices, ddof=1)),
        "percentile_5": float(np.percentile(terminal_prices, 5)),
        "percentile_95": float(np.percentile(terminal_prices, 95)),
        "probability_loss": float(np.mean(terminal_prices < starting_price)),
        "probability_above_target": float(np.mean(terminal_prices > target_price)),
        "target_price": float(target_price),
    }


def plot_stock_paths(
    paths: np.ndarray,
    output_path: str | Path,
    paths_to_plot: int = 100,
) -> None:
    """Plot a subset of simulated paths and save the figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_size = min(paths_to_plot, paths.shape[1])
    days = np.arange(paths.shape[0])

    plt.figure(figsize=(11, 6))
    plt.plot(days, paths[:, :sample_size], linewidth=0.8, alpha=0.55)
    plt.title("Monte Carlo Simulation of Stock Prices")
    plt.xlabel("Trading Days")
    plt.ylabel("Stock Price")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_terminal_distribution(
    terminal_prices: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot a terminal stock-price histogram with key percentiles."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_price = np.mean(terminal_prices)
    percentile_5 = np.percentile(terminal_prices, 5)
    percentile_95 = np.percentile(terminal_prices, 95)

    plt.figure(figsize=(11, 6))
    plt.hist(terminal_prices, bins=60, color="#4C78A8", alpha=0.82, edgecolor="white")
    plt.axvline(mean_price, color="#F58518", linewidth=2, label=f"Mean: ${mean_price:,.2f}")
