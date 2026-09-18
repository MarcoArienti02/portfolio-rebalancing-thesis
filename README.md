# Portfolio Rebalancing Strategies and Dynamic Asset Allocation

This repository collects the Python code I developed for my MSc thesis in Economics and Finance at the University of Milano-Bicocca. The thesis replicates the rebalancing framework of Dichtl, Drobetz and Wambach (2014) and then pushes it in two directions: forward in time (the sample is extended from December 2011 to May 2026) and forward in complexity (the same rebalancing rules are applied to portfolios whose target weights are re-estimated every month, namely Global Minimum Variance and Risk Parity).

Throughout the project, transaction costs, turnover and statistical inference are kept explicit: every strategy comparison is run under realistic trading costs, and every performance difference is tested with a stationary bootstrap (plus a parametric Jobson–Korkie cross-check), so that no conclusion rests on point estimates alone.

## Research objective

The starting question is the one posed by Dichtl et al. (2014): does disciplined rebalancing add risk-adjusted value relative to buy-and-hold? On top of that, my thesis asks two follow-up questions:

- does that value survive outside the original 1982–2011 sample, in a period with structurally different bond markets?
- does it survive when the target weights themselves move over time, as in GMV and Risk Parity portfolios?

The evidence turns out to be conditional on the sample period and on the target structure, and the code in this repository is what produces that evidence.

## Main research components

- Replication of Dichtl et al. (2014): 60/40 stock–bond portfolios for the US, the UK and Germany; buy-and-hold, periodic, threshold and range rebalancing at monthly, quarterly and yearly frequency; ±3% no-trade bands; proportional costs of 10 bps on equity and 5 bps on bonds.
- Temporal extension: the same protocol applied to the post-2011 (Jan 2012 – May 2026) and full extended (Jan 1982 – May 2026) samples, with robustness checks on cost scenarios and on threshold/range rules.
- Descriptive regime diagnostics: bond returns, yield paths and rolling equity–bond correlations across sub-periods.
- Multi-asset extension: a nine-asset USD universe (three equity indices, three 10Y government bond indices, gold, real estate, commodities) with GMV and Risk Parity (equal risk contribution) targets re-estimated monthly on a 60-month rolling covariance window, under long-only, fully-invested, no-leverage constraints.
- Inference: Politis–Romano stationary bootstrap with pairwise resampling (1,000 replications × 100 paths) and percentile confidence intervals on performance differences; for dynamic targets the bootstrap is conditioned on the historically estimated target sequence.
- Parametric cross-check: Jobson–Korkie test with Memmel correction for Sharpe-ratio differences.
- Synthetic constant-maturity government-bond total-return series built from yields and modified duration, validated against an external benchmark (correlations above 0.99 on the common sample).
- Performance metrics: Sharpe, Sortino and Omega ratios, plus annualized return, volatility, turnover and drawdown statistics.

## Headline results

- On 1982–2011 the replication confirms the original paper: all rebalancing rules outperform buy-and-hold at the 1% level across countries, horizons and metrics.
- On 2012–2026 the advantage largely disappears and, at the 10-year horizon, often reverses in favour of buy-and-hold; cost-scenario robustness shows that trading frictions are not the driver.
- With dynamic targets, Risk Parity behaves qualitatively like the 60/40 (favourable in-sample, unfavourable post-2011), while GMV rebalancing proves fragile throughout; conservative rules (infrequent checks, wide bands) cope best with estimation noise.

## Python implementation

The codebase separates the reusable research logic (`src/`) from the notebooks that run the analyses; the private data-preparation workflow is not included. The library provides:

- alignment of heterogeneous monthly series and return calculation;
- constant-maturity bond total-return approximation from yields and duration;
- simulation of two-asset and multi-asset rebalancing strategies;
- asset-specific transaction costs and portfolio-turnover accounting;
- risk-adjusted performance and drawdown statistics;
- look-ahead-free rolling estimation of GMV and Risk Parity targets; and
- bootstrap confidence intervals and parametric significance tests.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_bond_total_return_validation.ipynb
│   ├── 02_rolling_window_rebalancing.ipynb
│   ├── 03_stationary_bootstrap.ipynb
│   ├── 04_dynamic_allocation_gmv_risk_parity.ipynb
│   └── 05_sharpe_ratio_significance_tests.ipynb
└── src/
    ├── bootstrap/
    ├── data/
    ├── metrics/
    ├── optimization/
    └── portfolio/
```

The notebooks are a curated subset of the thesis workflow. Their saved outputs have been removed to avoid redistributing tables or figures derived from licensed data.

## Notebook guide

| Notebook | Focus |
|---|---|
| `01_bond_total_return_validation.ipynb` | Duration-based construction of government-bond total-return indices and validation against an external benchmark |
| `02_rolling_window_rebalancing.ipynb` | Five- and ten-year rolling evaluation of the 60/40 rebalancing rules on the original sample |
| `03_stationary_bootstrap.ipynb` | Politis–Romano stationary bootstrap and percentile confidence intervals on strategy differences |
| `04_dynamic_allocation_gmv_risk_parity.ipynb` | Rolling GMV and Risk Parity targets, multi-asset rebalancing backtests and bootstrap |
| `05_sharpe_ratio_significance_tests.ipynb` | Jobson–Korkie tests with Memmel correction for Sharpe-ratio differences |

## Data

The thesis uses Bloomberg and other licensed financial data. Those raw and processed datasets are **not included and may not be redistributed**. The public code therefore documents the expected schemas in [`data/README.md`](data/README.md), and all notebook outputs are cleared.

The repository does not fabricate substitute thesis results: anyone with appropriate data access can drop compatible files under `data/processed/` and run the notebooks locally.


## Academic context

- MSc in Economics and Finance, University of Milano-Bicocca
- Thesis title: *Il valore del ribilanciamento di portafoglio: dai pesi target statici a quelli dinamici basati sul rischio*
- Expected graduation: November 2026

## Status

The thesis is in its final revision stage. This repository is a curated, public-facing code portfolio and not a complete copy of the private research workspace.

## Disclaimer

This repository is provided for academic and educational purposes only. It does not constitute investment advice, a recommendation, or an offer to buy or sell financial instruments.
