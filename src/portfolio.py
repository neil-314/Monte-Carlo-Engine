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

