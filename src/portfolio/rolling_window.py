import pandas as pd

from src.metrics.performance_metrics import annualized_return, annualized_volatility, sharpe_ratio
from src.portfolio.rebalancing import run_rebalancing_strategy
# Build rolling windows of a fixed number of years.
def yearly_rolling_windows(df: pd.DataFrame, horizon_years: int, date_col: str = "Date") -> list[pd.DataFrame]:
    df = df.sort_values(date_col).reset_index(drop=True)
    start_year, end_year = df[date_col].dt.year.min(), df[date_col].dt.year.max() - horizon_years + 1

    # The paper uses January-December windows: 5 years = 60 months.
    return [
        w.reset_index(drop=True)
        for year in range(start_year, end_year + 1)
        for w in [df[df[date_col].between(f"{year}-01-31", f"{year + horizon_years - 1}-12-31")]]
        if len(w) == horizon_years * 12
    ]


def evaluate_rebalancing_window(
    df: pd.DataFrame,
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
        df[stock_col],
        df[bond_col],
        df["Date"],
        target_stock_w=target_stock_w,
        frequency=frequency,
        threshold=threshold,
        stock_cost=stock_cost,
        bond_cost=bond_cost,
    )
    merged = res.merge(df[["Date", rf_col]], on="Date", how="left")

    return {
        "Ann_Return": annualized_return(merged["Portfolio_Return"]),
        "Ann_Volatility": annualized_volatility(merged["Portfolio_Return"]),
        "Sharpe": sharpe_ratio(merged["Portfolio_Return"], merged[rf_col]),
        "Turnover": merged["Turnover"].sum(),
        "Transaction_Cost": merged["Transaction_Cost"].sum(),
    }

# Apply every rebalancing strategy to each rolling window.
def rolling_rebalancing_results(
    df: pd.DataFrame,
    horizon_years: int,
    strategies: dict[str, dict],
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for window in yearly_rolling_windows(df, horizon_years):
        start, end = window["Date"].min(), window["Date"].max()
        for name, params in strategies.items():
            rows.append(
                {
                    "Window_Start": start,
                    "Window_End": end,
                    "Horizon": horizon_years,
                    "Strategy": name,
                    **evaluate_rebalancing_window(window, **params, **kwargs),
                }
            )

    return pd.DataFrame(rows)
