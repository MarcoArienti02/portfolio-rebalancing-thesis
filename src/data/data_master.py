from pathlib import Path
import pandas as pd
from src.data.bond_returns import build_bond_total_return_index

PathLike = str | Path
START_DATE = pd.Timestamp("1982-01-31")
DEM_PER_EUR = 1.95583

COUNTRIES = {
    "US": {
        "equity_level": "USA_Equity_TR",
        "yield_cols": {5: "US_5Y_Yield", 10: "US_10Y_Yield"},
        "benchmark_10y_yield": "US_10Y_Benchmark_Yield",
        "rf_col": "US_3M_RF",
        "coupon_freq": 2,
    },
    "UK": {
        "equity_level": "UK_Equity_TR",
        "yield_cols": {5: "UK_5Y_Yield", 10: "UK_10Y_Yield"},
        "benchmark_10y_yield": "UK_10Y_Benchmark_Yield",
        "rf_col": "UK_RF",
        "coupon_freq": 1,
    },
    "DE": {
        "equity_level": "GER_Equity_Local_TR",
        "yield_cols": {5: "GER_5Y_Yield_BBK", 10: "GER_10Y_Yield_BBK"},
        "benchmark_10y_yield": "DE_10Y_Benchmark_Yield",
        "rf_col": "DE_RF",
        "coupon_freq": 1,
    },
}

def read_interim_table(interim_dir: PathLike, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(interim_dir) / f"{name}.csv", parse_dates=["Date"])

# Simple returns without implicit forward-filling.
def simple_returns(levels: pd.Series) -> pd.Series:
    # Avoid creating artificial returns across missing level observations.
    return levels.pct_change(fill_method=None)

# Convert an annual risk-free rate in percent to an approximate monthly return.
def monthly_rf_return(annual_yield_percent: pd.Series) -> pd.Series:
    return annual_yield_percent / 100 / 12
# Build equity series, including Germany's USD-denominated total-return series.
def build_equity_series(equity: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    out = equity.merge(fx, on="Date", how="left").rename(columns={"GER_Equity_TR": "GER_Equity_USD_TR"})

    # Convert German equity from USD to a continuous EUR-equivalent local series.
    out["GER_USD_per_Local"] = (out["DEMUSD"] * DEM_PER_EUR).where(
        out["Date"] < pd.Timestamp("1999-01-31"),
        out["EURUSD"],
    )
    out["GER_Equity_Local_TR"] = out["GER_Equity_USD_TR"] / out["GER_USD_per_Local"]

    return out

def build_risk_free_series(rf: pd.DataFrame) -> pd.DataFrame:
    out = rf[["Date", "US_3M_RF"]].copy()

    # UK: historical Treasury Bills, followed by SONIA when T-Bills end.
    out["UK_RF"] = rf["UK_3M_TBill_BOE"].combine_first(rf["UK_SONIA"])

    # Germany: pre-euro German proxy followed by EURIBOR in the euro period.
    out["DE_RF"] = rf["GER_3M_RF_FRED"].combine_first(rf["EURIBOR_3M"]).combine_first(rf["EUR_3M_RF_BBG"])
    return out


def add_10y_benchmark_yields(yields: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    out = yields.copy()
    bench = benchmark[["Date", *[cfg["benchmark_10y_yield"] for cfg in COUNTRIES.values()]]]
    out = out.merge(bench, on="Date", how="left")

    # Use the benchmark only to fill the initial gap in US 10Y yields.
    for cfg in COUNTRIES.values():
        col = cfg["yield_cols"][10]
        first_date = out.loc[out[col].notna(), "Date"].min()
        mask = out[col].isna() & out["Date"].between(START_DATE, first_date, inclusive="left")
        out.loc[mask, col] = out.loc[mask, cfg["benchmark_10y_yield"]]

    return out.drop(columns=[cfg["benchmark_10y_yield"] for cfg in COUNTRIES.values()])


def build_main_yield_series(yields: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    out = yields.copy()

    # UK: use the Bank of England historically, then the licensed source after coverage ends.
    out["UK_5Y_Yield"] = out["UK_5Y_Yield_BOE"].combine_first(out["UK_5Y_Yield_BBG"])
    out["UK_10Y_Yield"] = out["UK_10Y_Yield_BOE"].combine_first(out["UK_10Y_Yield_BBG"])

    return add_10y_benchmark_yields(out, benchmark)

def build_bond_series(yields: pd.DataFrame) -> dict[tuple[str, int], pd.DataFrame]:
    return {
        (country, maturity): build_bond_total_return_index(
            yields,
            yield_col,
            maturity=maturity,
            coupon_freq=cfg["coupon_freq"],
            index_name=f"{country}_Bond_{maturity}Y",
        )
        for country, cfg in COUNTRIES.items()
        for maturity, yield_col in cfg["yield_cols"].items()
    }

def build_master_levels(interim_dir: PathLike) -> pd.DataFrame:
    fx = read_interim_table(interim_dir, "cleaned_fx")
    equity = build_equity_series(read_interim_table(interim_dir, "cleaned_equity"), fx)
    alternatives = read_interim_table(interim_dir, "cleaned_alternatives")
    yields = build_main_yield_series(
        read_interim_table(interim_dir, "cleaned_yields"),
        read_interim_table(interim_dir, "cleaned_10y_bond_tr_benchmarks"),
    )
    rf = build_risk_free_series(read_interim_table(interim_dir, "cleaned_risk_free"))
    bonds = build_bond_series(yields)

    master = equity.merge(rf, on="Date", how="outer").merge(alternatives, on="Date", how="outer")

    # Add synthetic bond indices and the yields used to construct them.
    for (country, maturity), bond in bonds.items():
        cols = ["Date", f"{country}_Bond_{maturity}Y_Yield", f"{country}_Bond_{maturity}Y_TR_Index"]
        master = master.merge(bond[cols], on="Date", how="outer")

    return master.sort_values("Date").reset_index(drop=True)

def build_master_returns(levels: pd.DataFrame) -> pd.DataFrame:
    out = levels[["Date"]].copy()

    for country, cfg in COUNTRIES.items():
        out[f"{country}_Equity_Return"] = simple_returns(levels[cfg["equity_level"]])
        for maturity in cfg["yield_cols"]:
            out[f"{country}_Bond_{maturity}Y_Return"] = simple_returns(levels[f"{country}_Bond_{maturity}Y_TR_Index"])
        out[f"{country}_RF_Return"] = monthly_rf_return(levels[cfg["rf_col"]])

    # Additional FX and alternative-asset returns used in extensions.
    for col in ["GBPUSD", "EURUSD", "DEMUSD", "GER_USD_per_Local", "Gold", "Real_Estate_TR", "Commodities_TR"]:
        out[f"{col}_Return"] = simple_returns(levels[col])

    return out

def build_country_dataset(levels: pd.DataFrame, returns: pd.DataFrame, country: str) -> pd.DataFrame:
    cfg = COUNTRIES[country]
    out = levels[
        [
            "Date",
            cfg["equity_level"],
            f"{country}_Bond_5Y_TR_Index",
            f"{country}_Bond_10Y_TR_Index",
            cfg["rf_col"],
            f"{country}_Bond_5Y_Yield",
            f"{country}_Bond_10Y_Yield",
        ]
    ].copy()

    out.columns = [
        "Date",
        "Equity_TR_Index",
        "Bond_5Y_TR_Index",
        "Bond_10Y_TR_Index",
        "RF_Annual_Yield",
        "Bond_5Y_Yield",
        "Bond_10Y_Yield",
    ]

    if country == "DE":
        out["Equity_USD_TR_Index"] = levels["GER_Equity_USD_TR"]
        out["FX_USD_per_Local"] = levels["GER_USD_per_Local"]

    out = out.merge(
        returns[
            [
                "Date",
                f"{country}_Equity_Return",
                f"{country}_Bond_5Y_Return",
                f"{country}_Bond_10Y_Return",
                f"{country}_RF_Return",
            ]
        ].rename(
            columns={
                f"{country}_Equity_Return": "Equity_Return",
                f"{country}_Bond_5Y_Return": "Bond_5Y_Return",
                f"{country}_Bond_10Y_Return": "Bond_10Y_Return",
                f"{country}_RF_Return": "RF_Return",
            }
        ),
        on="Date",
        how="left",
    )

    # Retain the first level observation; returns begin one month later by construction.
    return out.dropna(subset=["Equity_TR_Index", "Bond_10Y_TR_Index", "RF_Annual_Yield"]).reset_index(drop=True)

def save_processed_table(df: pd.DataFrame, path: PathLike) -> dict[str, Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path, xlsx_path = path.with_suffix(".csv"), path.with_suffix(".xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    # Apply basic workbook formatting for manual validation.
    from openpyxl import load_workbook

    wb = load_workbook(xlsx_path)
    ws = wb.active
    ws.freeze_panes = "A2"
    headers = {cell.column: cell.value for cell in ws[1]}
    for col in ws.columns:
        letter = col[0].column_letter
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col[:200]) + 2
        ws.column_dimensions[letter].width = min(max(width, 12), 28)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            header = headers[cell.column]
            if cell.column == 1:
                cell.number_format = "yyyy-mm-dd"
            elif isinstance(cell.value, float) and ("Return" in header or ("Bond_" in header and header.endswith("_Yield"))):
                cell.number_format = "0.00%"
            elif isinstance(cell.value, float):
                cell.number_format = "0.00"
    wb.save(xlsx_path)

    return {"csv": csv_path, "excel": xlsx_path}


def build_processed_datasets(interim_dir: PathLike, processed_dir: PathLike) -> dict[str, dict[str, Path]]:
    processed_dir = Path(processed_dir)
    levels = build_master_levels(interim_dir)
    returns = build_master_returns(levels)
    datasets = {
        "master_monthly_levels": levels,
        "master_monthly_returns": returns,
        **{
            f"{country.lower()}_data": build_country_dataset(levels, returns, country)
            for country in COUNTRIES
        },
    }

    return {
        name: save_processed_table(df, processed_dir / name)
        for name, df in datasets.items()
    }
