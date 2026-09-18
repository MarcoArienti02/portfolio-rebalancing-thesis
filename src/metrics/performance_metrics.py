import math

import numpy as np
import pandas as pd

def clean_returns(ret: pd.Series) -> pd.Series:
    return pd.to_numeric(ret, errors="coerce").dropna()

def log_returns(ret: pd.Series) -> pd.Series:
    return np.log1p(clean_returns(ret))

def cumulative_return_index(ret: pd.Series, base: float = 100) -> pd.Series:
    # Convert periodic returns into a cumulative wealth index.
    return base * (1 + clean_returns(ret)).cumprod()

# Annualized return from a monthly return series.
def annualized_return(ret: pd.Series, periods_per_year: int = 12, method: str = "compound") -> float:
    ret = clean_returns(ret)
    if ret.empty:
        return np.nan

    if method == "mean":
        return float(ret.mean() * periods_per_year)
    if method == "log":
        return float(log_returns(ret).mean() * periods_per_year)

    return float((1 + ret).prod() ** (periods_per_year / len(ret)) - 1)


def annualized_volatility(ret: pd.Series, periods_per_year: int = 12, use_log: bool = False) -> float:
    ret = log_returns(ret) if use_log else clean_returns(ret)
    return float(ret.std(ddof=1) * np.sqrt(periods_per_year))

def excess_returns(ret: pd.Series, rf: pd.Series | float = 0) -> pd.Series:
    # Align a risk-free series to returns; otherwise use the scalar value.
    if isinstance(rf, pd.Series):
        data = pd.concat([ret, rf], axis=1).dropna()
        return data.iloc[:, 0] - data.iloc[:, 1]

    return clean_returns(ret) - rf

# Annualized Sharpe ratio.
def sharpe_ratio(ret: pd.Series, rf: pd.Series | float = 0, periods_per_year: int = 12) -> float:
    exc = excess_returns(ret, rf)
    vol = exc.std(ddof=1)
    return np.nan if vol == 0 or pd.isna(vol) else float(exc.mean() / vol * np.sqrt(periods_per_year))


def _significance_stars(p_value: float) -> str:
    return "***" if p_value <= 0.01 else "**" if p_value <= 0.05 else "*" if p_value <= 0.10 else ""


def jobson_korkie_memmel_test(
    ret_1: pd.Series,
    ret_2: pd.Series,
    rf: pd.Series | float = 0,
    periods_per_year: int = 12,
) -> dict[str, float | str]:
    """Jobson-Korkie test with Memmel correction for two Sharpe ratios.

    This parametric robustness check complements the stationary bootstrap and
    assumes approximately normal returns with limited serial dependence.
    """
    if isinstance(rf, pd.Series):
        data = pd.concat([ret_1, ret_2, rf], axis=1).dropna()
        x, y = data.iloc[:, 0] - data.iloc[:, 2], data.iloc[:, 1] - data.iloc[:, 2]
    else:
        data = pd.concat([ret_1, ret_2], axis=1).dropna()
        x, y = data.iloc[:, 0] - rf, data.iloc[:, 1] - rf

    n_obs = len(data)
    std_x, std_y = x.std(ddof=1), y.std(ddof=1)
    if n_obs < 3 or std_x == 0 or std_y == 0:
        return {k: np.nan for k in ["n_obs", "sharpe_1", "sharpe_2", "sharpe_diff", "rho", "z_stat", "p_value"]} | {"significance": ""}

    sr_1, sr_2 = x.mean() / std_x, y.mean() / std_y
    rho = x.corr(y)
    var = 2 * (1 - rho) + 0.5 * (sr_1**2 + sr_2**2 - 2 * sr_1 * sr_2 * rho**2)
    z = np.nan if var <= 0 or pd.isna(var) else (sr_1 - sr_2) * np.sqrt(n_obs / var)
    p_value = np.nan if pd.isna(z) else math.erfc(abs(z) / np.sqrt(2))

    # Report annualized Sharpe ratios; the z-statistic uses monthly inputs.
    return {
        "n_obs": n_obs,
        "sharpe_1": float(sr_1 * np.sqrt(periods_per_year)),
        "sharpe_2": float(sr_2 * np.sqrt(periods_per_year)),
        "sharpe_diff": float((sr_1 - sr_2) * np.sqrt(periods_per_year)),
        "rho": float(rho),
        "z_stat": float(z),
        "p_value": float(p_value),
        "significance": "" if pd.isna(p_value) else _significance_stars(p_value),
    }

# Annualized downside deviation.
def downside_deviation(ret: pd.Series, target: pd.Series | float = 0, periods_per_year: int = 12) -> float:
    diff = excess_returns(ret, target)
    downside = diff.clip(upper=0)
    return float(np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year))

# Annualized Sortino ratio.
def sortino_ratio(ret: pd.Series, target: pd.Series | float = 0, periods_per_year: int = 12) -> float:
    diff = excess_returns(ret, target)
    downside = downside_deviation(ret, target, periods_per_year)
    return np.nan if downside == 0 or pd.isna(downside) else float(diff.mean() * periods_per_year / downside)


def omega_ratio(ret: pd.Series, threshold: pd.Series | float = 0) -> float:
    diff = excess_returns(ret, threshold)
    gains, losses = diff.clip(lower=0).sum(), -diff.clip(upper=0).sum()
    return np.nan if losses == 0 or pd.isna(losses) else float(gains / losses)


def max_drawdown(ret: pd.Series) -> float:
    wealth = cumulative_return_index(ret, base=1)
    return float((wealth / wealth.cummax() - 1).min())


def descriptive_stats(ret: pd.Series, periods_per_year: int = 12, use_log: bool = True) -> dict[str, float]:
    r = log_returns(ret) if use_log else clean_returns(ret)
    return {
        "mean_ann": float(r.mean() * periods_per_year),
        "vol_ann": float(r.std(ddof=1) * np.sqrt(periods_per_year)),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis() + 3),
        "min": float(r.min()),
        "max": float(r.max()),
    }


def performance_summary(
    ret: pd.Series,
    rf: pd.Series | float = 0,
    target: pd.Series | float = 0,
    periods_per_year: int = 12,
) -> dict[str, float]:
    # Compact summary used by notebooks and bootstrap simulations.
    return {
        "ann_return": annualized_return(ret, periods_per_year),
        "ann_vol": annualized_volatility(ret, periods_per_year),
        "sharpe": sharpe_ratio(ret, rf, periods_per_year),
        "sortino": sortino_ratio(ret, target, periods_per_year),
        "omega": omega_ratio(ret, target),
        "max_drawdown": max_drawdown(ret),
    }
