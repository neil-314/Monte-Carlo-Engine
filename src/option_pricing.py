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

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0.0)

    return float(np.exp(-r * T) * np.mean(payoffs))


def black_scholes_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Calculate the analytical Black-Scholes price for a European option."""
    _validate_option_type(option_type)
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")

    if sigma == 0:
        forward_price = S0 * np.exp(r * T)
        if option_type == "call":
            payoff = max(forward_price - K, 0.0)
        else:
            payoff = max(K - forward_price, 0.0)
        return float(np.exp(-r * T) * payoff)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))


def compare_option_prices(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    simulations: int,
    option_type: OptionType = "call",
    seed: int | None = None,
) -> dict[str, float]:
    """Compare Monte Carlo and Black-Scholes prices for one option."""
    mc_price = monte_carlo_option_price(S0, K, T, r, sigma, simulations, option_type, seed)
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type)
    absolute_difference = abs(mc_price - bs_price)
    percentage_error = absolute_difference / bs_price if bs_price != 0 else 0.0

    return {
        "monte_carlo_price": mc_price,
        "black_scholes_price": bs_price,
        "absolute_difference": absolute_difference,
        "percentage_error": percentage_error,
    }


def convergence_analysis(
    simulation_counts: list[int],
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    seed: int = 123,
) -> pd.DataFrame:
    """Run option pricing at several simulation counts and measure error."""
    rows = []
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type)

    for index, simulations in enumerate(simulation_counts):
        mc_price = monte_carlo_option_price(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            simulations=simulations,
            option_type=option_type,
            seed=seed + index,
        )
        absolute_error = abs(mc_price - bs_price)
        rows.append(
            {
                "simulations": simulations,
                "monte_carlo_price": mc_price,
                "black_scholes_price": bs_price,
                "absolute_error": absolute_error,
                "percentage_error": absolute_error / bs_price if bs_price != 0 else 0.0,
            }
        )

    return pd.DataFrame(rows)


def plot_convergence(convergence: pd.DataFrame, output_path: str | Path) -> None:
    """Plot Monte Carlo option convergence against Black-Scholes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (price_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    price_axis.plot(
        convergence["simulations"],
        convergence["monte_carlo_price"],
        color="#4C78A8",
        marker="o",
        linewidth=2,
        label="Monte Carlo price",
    )
    price_axis.axhline(
        convergence["black_scholes_price"].iloc[0],
        color="#F58518",
        linestyle="--",
        linewidth=2,
        label="Black-Scholes price",
    )
    price_axis.set_xscale("log")
    price_axis.set_ylabel("Option Price")
    price_axis.set_title("Monte Carlo Option Pricing Convergence")
    price_axis.grid(True, alpha=0.25)
    price_axis.legend()

    error_axis.plot(
        convergence["simulations"],
        convergence["absolute_error"],
        color="#E45756",
        marker="o",
        linewidth=2,
        label="Absolute error",
    )
    error_axis.set_xscale("log")
    error_axis.set_xlabel("Number of Simulations")
    error_axis.set_ylabel("Error")
    error_axis.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def format_option_report(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    simulations: int,
    call_comparison: dict[str, float],
    put_comparison: dict[str, float],
) -> str:
    """Format option pricing results for console output."""
    return f"""
EUROPEAN OPTION PRICING

Stock Price: ${S0:,.2f}
Strike:      ${K:,.2f}
Expiration:  {T:g} year
Volatility:  {sigma:.0%}
Risk-Free:   {r:.0%}
Scenarios:   {simulations:,}

Monte Carlo Call:    ${call_comparison["monte_carlo_price"]:,.2f}
Black-Scholes Call:  ${call_comparison["black_scholes_price"]:,.2f}
Call Abs. Diff.:     ${call_comparison["absolute_difference"]:,.2f}
Call % Error:         {call_comparison["percentage_error"]:.2%}

Monte Carlo Put:     ${put_comparison["monte_carlo_price"]:,.2f}
Black-Scholes Put:   ${put_comparison["black_scholes_price"]:,.2f}
Put Abs. Diff.:      ${put_comparison["absolute_difference"]:,.2f}
Put % Error:          {put_comparison["percentage_error"]:.2%}
""".strip()


def main() -> None:
    S0 = 100
    K = 105
    T = 1
    r = 0.04
    sigma = 0.20
    simulations = 100_000

    call_comparison = compare_option_prices(S0, K, T, r, sigma, simulations, "call", seed=7)
    put_comparison = compare_option_prices(S0, K, T, r, sigma, simulations, "put", seed=11)
    print(format_option_report(S0, K, T, r, sigma, simulations, call_comparison, put_comparison))

    convergence = convergence_analysis(
        [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000],
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
    )
    plot_convergence(convergence, "results/option_convergence.png")
