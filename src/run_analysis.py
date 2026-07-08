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
