import numpy as np

from option_pricing import (
    compare_option_prices,
    convergence_analysis,
    format_option_report,
    plot_convergence,
)
from portfolio import (
    format_portfolio_report,
    plot_portfolio_distribution,
    simulate_correlated_portfolio,
)
from risk_metrics import portfolio_risk_report
from simulation import (
    format_terminal_report,
    plot_stock_paths,
    plot_terminal_distribution,
    simulate_gbm,
    terminal_price_statistics,
)


def run_stock_simulation() -> None:
    S0 = 100
    mu = 0.08
    sigma = 0.20
    T = 1
    steps = 252
    simulations = 10_000
    target_price = 120

    paths = simulate_gbm(S0, mu, sigma, T, steps, simulations, seed=42)
    terminal_prices = paths[-1]
    terminal_stats = terminal_price_statistics(terminal_prices, S0, target_price)

    plot_stock_paths(paths, "results/stock_paths.png")
    plot_terminal_distribution(terminal_prices, "results/terminal_distribution.png")

    print(format_terminal_report(terminal_stats))
    print()


def run_option_pricing() -> None:
    S0 = 100
    K = 105
    T = 1
    r = 0.04
    sigma = 0.20
    simulations = 100_000

    call_comparison = compare_option_prices(S0, K, T, r, sigma, simulations, "call", seed=7)
    put_comparison = compare_option_prices(S0, K, T, r, sigma, simulations, "put", seed=11)

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

    print(format_option_report(S0, K, T, r, sigma, simulations, call_comparison, put_comparison))
    print()


def run_portfolio_simulation() -> None:
    initial_value = 100_000
    weights = np.array([0.40, 0.30, 0.20, 0.10])
    expected_returns = np.array([0.08, 0.10, 0.12, 0.06])
    volatilities = np.array([0.20, 0.25, 0.30, 0.15])
    correlation_matrix = np.array(
        [
            [1.00, 0.70, 0.55, 0.30],
            [0.70, 1.00, 0.65, 0.25],
            [0.55, 0.65, 1.00, 0.20],
            [0.30, 0.25, 0.20, 1.00],
        ]
    )

    portfolio = simulate_correlated_portfolio(
        initial_value=initial_value,
        weights=weights,
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        simulations=100_000,
        seed=21,
        asset_names=["Asset A", "Asset B", "Asset C", "Asset D"],
    )
    report = portfolio_risk_report(
        portfolio["Portfolio Value"].to_numpy(),
        portfolio["Portfolio Return"].to_numpy(),
        initial_value=initial_value,
    )
    plot_portfolio_distribution(portfolio["Portfolio Return"].to_numpy(), "results/portfolio_distribution.png")

    print(format_portfolio_report(report))
    print()


def main() -> None:
    run_stock_simulation()
    run_option_pricing()
    run_portfolio_simulation()
    print("Charts saved:")
    print("- results/stock_paths.png")
