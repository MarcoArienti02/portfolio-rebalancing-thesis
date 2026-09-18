import numpy as np
import pandas as pd

from src.metrics.performance_metrics import performance_summary
from src.portfolio.rebalancing import run_rebalancing_strategy


FREQ_MONTHS = {"M": 1, "Q": 3, "Y": 12, "monthly": 1, "quarterly": 3, "yearly": 12}
PERF_METRICS = ["ann_return", "ann_vol", "sharpe", "sortino", "omega", "max_drawdown"]


def stationary_bootstrap_indices(
    n_obs: int,
    sample_size: int,
    avg_block_length: float = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate Politis-Romano (1994) stationary-bootstrap indices.

    A common index path is applied to all columns to preserve contemporaneous
    dependence among equity, bond, and risk-free returns.
    """
    rng = np.random.default_rng(rng)
    p = 1 / avg_block_length
    idx = np.empty(sample_size, dtype=int)
    idx[0] = rng.integers(n_obs)
    # Continue the current block or restart from a random observation.
    for t in range(1, sample_size):
        idx[t] = rng.integers(n_obs) if rng.random() < p else (idx[t - 1] + 1) % n_obs

    return idx

# Build a bootstrap sample from generated indices.
def stationary_bootstrap_sample(
    df: pd.DataFrame,
    sample_size: int,
    avg_block_length: float = 2,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    idx = stationary_bootstrap_indices(len(df), sample_size, avg_block_length, rng)
    return df.iloc[idx].reset_index(drop=True)

# Evaluate one strategy on a simulated path.
def evaluate_strategy_on_path(
    path: pd.DataFrame,
    strategy: str,
    frequency: str = "Q",
    stock_col: str = "Equity_Return",
    bond_col: str = "Bond_10Y_Return",
    rf_col: str = "RF_Return",
    target_stock_w: float = 0.60,
    threshold: float = 0.03,
    stock_cost: float = 0.001,
    bond_cost: float = 0.0005,
) -> dict[str, float]:
    res = run_rebalancing_strategy(
        strategy,
        path[stock_col],
        path[bond_col],
        dates=None,
        target_stock_w=target_stock_w,
        frequency=frequency,
        threshold=threshold,
        stock_cost=stock_cost,
        bond_cost=bond_cost,
    )
    rf = path[rf_col].reset_index(drop=True).iloc[: len(res)]

    # Apply the shared metric definitions to simulated returns.
    return performance_summary(res["Portfolio_Return"], rf=rf, target=0)


def _portfolio_returns_array(
    stock_ret: np.ndarray,
    bond_ret: np.ndarray,
    strategy: str,
    frequency: str = "Q",
    target_stock_w: float = 0.60,
    threshold: float = 0.03,
    stock_cost: float = 0.001,
    bond_cost: float = 0.0005,
) -> pd.Series:
    stock_w, wealth = target_stock_w, 1.0
    ret = np.empty(len(stock_ret))

    for i, (rs, rb) in enumerate(zip(stock_ret, bond_ret)):
        start_wealth = wealth
        stock_val = wealth * stock_w * (1 + rs)
        bond_val = wealth * (1 - stock_w) * (1 + rb)
        pre_wealth = stock_val + bond_val
        pre_stock_w = stock_val / pre_wealth
        post_stock_w = pre_stock_w

        # Bootstrap paths have simulated month positions rather than calendar dates.
        is_reb_date = strategy != "buy_and_hold" and (i + 1) % FREQ_MONTHS[frequency] == 0
        if strategy == "periodic" and is_reb_date:
            post_stock_w = target_stock_w
        elif strategy in ["threshold", "range"] and is_reb_date:
            lower, upper = target_stock_w - threshold, target_stock_w + threshold
            if pre_stock_w < lower or pre_stock_w > upper:
                post_stock_w = target_stock_w if strategy == "threshold" else lower if pre_stock_w < lower else upper

        trade = abs(post_stock_w - pre_stock_w)
        wealth = pre_wealth * (1 - trade * (stock_cost + bond_cost))
        stock_w = post_stock_w
        ret[i] = wealth / start_wealth - 1

    return pd.Series(ret)


def evaluate_strategy_arrays(
    stock_ret: np.ndarray,
    bond_ret: np.ndarray,
    rf: np.ndarray,
    strategy: str,
    frequency: str = "Q",
    target_stock_w: float = 0.60,
    threshold: float = 0.03,
    stock_cost: float = 0.001,
    bond_cost: float = 0.0005,
) -> dict[str, float]:
    ret = _portfolio_returns_array(stock_ret, bond_ret, strategy, frequency, target_stock_w, threshold, stock_cost, bond_cost)
    return performance_summary(ret, rf=pd.Series(rf), target=0)


def _performance_summary_array(ret: np.ndarray, rf: np.ndarray, metrics: tuple[str, ...]) -> dict[str, float]:
    out = {}

    if "ann_return" in metrics:
        out["ann_return"] = float((1 + ret).prod() ** (12 / len(ret)) - 1)
    if "ann_vol" in metrics:
        out["ann_vol"] = float(ret.std(ddof=1) * np.sqrt(12))
    if "sharpe" in metrics:
        exc = ret - rf
        vol = exc.std(ddof=1)
        out["sharpe"] = np.nan if vol == 0 or np.isnan(vol) else float(exc.mean() / vol * np.sqrt(12))
    if "sortino" in metrics:
        # The paper uses a zero target for Sortino.
        downside = np.sqrt((np.clip(ret, None, 0) ** 2).mean()) * np.sqrt(12)
        out["sortino"] = np.nan if downside == 0 or np.isnan(downside) else float(ret.mean() * 12 / downside)
    if "omega" in metrics:
        # Omega compares gains and losses relative to zero.
        gains, losses = np.clip(ret, 0, None).sum(), -np.clip(ret, None, 0).sum()
        out["omega"] = np.nan if losses == 0 or np.isnan(losses) else float(gains / losses)
    if "max_drawdown" in metrics:
        wealth = np.cumprod(1 + ret)
        out["max_drawdown"] = float((wealth / np.maximum.accumulate(wealth) - 1).min())

    return out


def bootstrap_strategy_performance(
    df: pd.DataFrame,
    strategies: dict[str, dict],
    horizon_years: int,
    n_sims: int = 1000,
    paths_per_sim: int = 100,
    avg_block_length: float = 2,
    random_state: int | None = None,
    required_cols: tuple[str, ...] = ("Equity_Return", "Bond_10Y_Return", "RF_Return"),
    **kwargs,
) -> pd.DataFrame:
    data = df.dropna(subset=list(required_cols)).reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    sample_size = horizon_years * 12
    rows = []

    for sim in range(n_sims):
        for path_id in range(paths_per_sim):
            path = stationary_bootstrap_sample(data, sample_size, avg_block_length, rng)
            for name, params in strategies.items():
                rows.append(
                    {
                        "Simulation": sim,
                        "Path": path_id,
                        "Horizon": horizon_years,
                        "Strategy": name,
                        **evaluate_strategy_on_path(path, **(params | kwargs)),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_average_strategy_performance(
    df: pd.DataFrame,
    strategies: dict[str, dict],
    horizon_years: int,
    n_sims: int = 1000,
    paths_per_sim: int = 100,
    avg_block_length: float = 2,
    random_state: int | None = None,
    required_cols: tuple[str, ...] = ("Equity_Return", "Bond_10Y_Return", "RF_Return"),
    metrics: tuple[str, ...] = tuple(PERF_METRICS),
    **kwargs,
) -> pd.DataFrame:
    data = df.dropna(subset=list(required_cols)).reset_index(drop=True)
    arr = data.loc[:, list(required_cols)].to_numpy()
    rng = np.random.default_rng(random_state)
    rows = []

    for sim in range(n_sims):
        sums = {name: dict.fromkeys(metrics, 0.0) for name in strategies}
        for _ in range(paths_per_sim):
            path = arr[stationary_bootstrap_indices(len(arr), horizon_years * 12, avg_block_length, rng)]
            for name, params in strategies.items():
                ret = _portfolio_returns_array(path[:, 0], path[:, 1], **(params | kwargs)).to_numpy()
                res = _performance_summary_array(ret, path[:, 2], metrics)
                for metric in metrics:
                    sums[name][metric] += res[metric]

        # Average paths within each simulation, following the paper's design.
        rows += [
            {
                "Simulation": sim,
                "Horizon": horizon_years,
                "Strategy": name,
                **{metric: sums[name][metric] / paths_per_sim for metric in metrics},
            }
            for name in strategies
        ]

    return pd.DataFrame(rows)

# Average strategy performance within each bootstrap simulation.
def average_performance_by_simulation(perf: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in PERF_METRICS if metric in perf.columns]
    return perf.groupby(["Simulation", "Horizon", "Strategy"], as_index=False)[metrics].mean()

# Compute pairwise strategy differences for a selected metric.
def strategy_differences(
    perf: pd.DataFrame,
    pairs: dict[str, tuple[str, str]],
    metric: str = "sharpe",
) -> pd.DataFrame:
    avg = average_performance_by_simulation(perf)
    wide = avg.pivot(index=["Simulation", "Horizon"], columns="Strategy", values=metric)

    # Difference A-B supports rebalancing versus buy-and-hold comparisons.
    return pd.concat(
        [
            (wide[a] - wide[b]).rename(metric).reset_index().assign(Comparison=name)
            for name, (a, b) in pairs.items()
        ],
        ignore_index=True,
    )

# Percentile confidence intervals for strategy differences.
def percentile_confidence_intervals(
    diffs: pd.DataFrame,
    metric: str = "sharpe",
    alpha: float = 0.01,
) -> pd.DataFrame:
    q = diffs.groupby(["Horizon", "Comparison"])[metric].quantile([alpha / 2, 1 - alpha / 2]).unstack()
    q.columns = ["CI_Lower", "CI_Upper"]
    return q.reset_index()
# A confidence interval excluding zero indicates statistical significance.

# Final confidence summary by horizon and comparison.
def bootstrap_confidence_report(diffs: pd.DataFrame, metric: str = "sharpe") -> pd.DataFrame:
    rows = []
    for (horizon, comparison), data in diffs.groupby(["Horizon", "Comparison"]):
        vals = data[metric].dropna()
        bounds = {
            alpha: vals.quantile([alpha / 2, 1 - alpha / 2]).to_numpy()
            for alpha in [0.01, 0.05, 0.10]
        }
        stars = "***" if np.prod(bounds[0.01]) > 0 else "**" if np.prod(bounds[0.05]) > 0 else "*" if np.prod(bounds[0.10]) > 0 else ""
        alpha = 0.01 if stars == "***" else 0.05 if stars == "**" else 0.10

        # If not significant at 10%, still report the 90% interval.
        rows.append(
            {
                "Horizon": horizon,
                "Comparison": comparison,
                "CI_Lower": bounds[alpha][0],
                "CI_Upper": bounds[alpha][1],
                "Significance": stars,
            }
        )

    return pd.DataFrame(rows)


def summarize_bootstrap_performance(perf: pd.DataFrame) -> pd.DataFrame:
    avg = average_performance_by_simulation(perf)
    metrics = [metric for metric in PERF_METRICS if metric in avg.columns]
    return avg.groupby(["Horizon", "Strategy"], as_index=False)[metrics].mean()
