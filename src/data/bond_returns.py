from pathlib import Path

import numpy as np
import pandas as pd

PathLike = str | Path


def _check_yield_scale(y: pd.Series, value_in_percent: bool) -> None:
    avg = y.dropna().abs().mean()

    # Scale guard: 5% must be represented as 5 when value_in_percent=True.
    if value_in_percent and avg < 0.05:
        raise ValueError("Yields appear to be decimal values, but value_in_percent=True.")
    if not value_in_percent and avg > 1:
        raise ValueError("Yields appear to be percentages, but value_in_percent=False.")


def modified_duration_par_bond(
    y: float,
    maturity: float,
    coupon_freq: int = 2,
    face: float = 100,
    allow_negative_yields: bool = True,
) -> float:
    """
    Calculate the modified duration of a par bond using coupon rate = initial yield.

    Practical convention:
    - US Treasuries typically use coupon_freq=2.
    - UK Gilts and German Bunds typically use coupon_freq=1.
    - Negative yields are allowed as long as the periodic discount factor remains positive.
      This avoids arbitrarily clipping yields during the negative-rate period.
    """
    if y < 0 and not allow_negative_yields:
        raise ValueError("Negative yield found while allow_negative_yields=False.")

    period_y = y / coupon_freq
    if 1 + period_y <= 0:
        raise ValueError("Invalid discount factor: 1 + y / coupon_freq <= 0.")

    n = int(round(maturity * coupon_freq))
    times = np.arange(1, n + 1) / coupon_freq
    cash_flows = np.full(n, y / coupon_freq * face)
    cash_flows[-1] += face

    pv = cash_flows / (1 + period_y) ** np.arange(1, n + 1)
    macaulay = (times * pv).sum() / pv.sum()

    return macaulay / (1 + period_y)


def total_return_index(returns: pd.Series, base: float = 100) -> pd.Series:
    # The first return is undefined by construction; the index still starts at the first date.
    return base * (1 + returns.fillna(0)).cumprod()


def compute_bond_total_returns(
    df: pd.DataFrame,
    yield_col: str,
    maturity: float,
    coupon_freq: int = 2,
    date_col: str = "Date",
    value_in_percent: bool = True,
    allow_negative_yields: bool = True,
) -> pd.DataFrame:
    """
    Build a monthly constant-maturity bond total-return proxy from 5Y/10Y yields.

    Method:
    bond_return_t = y_{t-1}/12 - modified_duration_{t-1} * (y_t - y_{t-1})

    Methodological notes:
    - the method constructs a constant-maturity total-return proxy;
    - it does not exactly reproduce a Bloomberg or Datastream bond index;
    - the income component is approximated by y_{t-1}/12;
    - the price component uses modified duration based on information available at month start;
    - the series should be validated against an external benchmark.
    """
    out = df[[date_col, yield_col]].rename(columns={date_col: "Date"}).copy()
    out["Yield"] = pd.to_numeric(out[yield_col], errors="coerce")
    out = out.dropna(subset=["Date", "Yield"])[["Date", "Yield"]]
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    _check_yield_scale(out["Yield"], value_in_percent)
    out["Yield"] = out["Yield"] / 100 if value_in_percent else out["Yield"]

    # Use only information available at the start of the month.
    y_lag = out["Yield"].shift(1)
    dy = out["Yield"].diff()
    dur = y_lag.map(
        lambda y: modified_duration_par_bond(
            y,
            maturity=maturity,
            coupon_freq=coupon_freq,
            allow_negative_yields=allow_negative_yields,
        )
        if pd.notna(y)
        else np.nan
    )

    out["Income_Return"] = y_lag / 12
    out["Price_Return"] = -dur * dy
    out["Bond_Return"] = out["Income_Return"] + out["Price_Return"]
    out["Bond_TR_Index"] = total_return_index(out["Bond_Return"])
    out["Modified_Duration"] = dur

    return out


def build_bond_total_return_index(
    df: pd.DataFrame,
    yield_col: str,
    maturity: float,
    coupon_freq: int = 2,
    date_col: str = "Date",
    value_in_percent: bool = True,
    index_name: str | None = None,
) -> pd.DataFrame:
    res = compute_bond_total_returns(
        df=df,
        yield_col=yield_col,
        maturity=maturity,
        coupon_freq=coupon_freq,
        date_col=date_col,
        value_in_percent=value_in_percent,
    )

    if index_name is not None:
        res = res.rename(
            columns={
                "Yield": f"{index_name}_Yield",
                "Bond_Return": f"{index_name}_Return",
                "Bond_TR_Index": f"{index_name}_TR_Index",
                "Modified_Duration": f"{index_name}_Modified_Duration",
            }
        )

    return res


def validate_against_benchmark(
    synthetic: pd.DataFrame,
    benchmark: pd.DataFrame,
    synthetic_return_col: str = "Bond_Return",
    benchmark_return_col: str = "Benchmark_Return",
    date_col: str = "Date",
    benchmark_in_percent: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    bench = benchmark[[date_col, benchmark_return_col]].copy()
    bench[benchmark_return_col] = bench[benchmark_return_col] / 100 if benchmark_in_percent else bench[benchmark_return_col]

    comp = synthetic[[date_col, synthetic_return_col]].merge(bench, on=date_col, how="inner")
    comp = comp.dropna(subset=[synthetic_return_col, benchmark_return_col]).reset_index(drop=True)
    comp["Return_Diff"] = comp[synthetic_return_col] - comp[benchmark_return_col]

    # Compact diagnostics for agreement with the external benchmark.
    stats = {
        "n_obs": len(comp),
        "corr": float(comp[synthetic_return_col].corr(comp[benchmark_return_col])),
        "mae": float(comp["Return_Diff"].abs().mean()),
        "rmse": float(np.sqrt((comp["Return_Diff"] ** 2).mean())),
        "mean_synthetic": float(comp[synthetic_return_col].mean()),
        "mean_benchmark": float(comp[benchmark_return_col].mean()),
        "vol_synthetic": float(comp[synthetic_return_col].std()),
        "vol_benchmark": float(comp[benchmark_return_col].std()),
    }

    return comp, stats
