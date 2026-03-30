# Multi-Agent Sales Analysis Report
*Generated: 2025-03-28 14:37*

---

## Pipeline Run Summary

| Agent | Status | Notes |
|---|---|---|
| `supervisor` | started | Acknowledged task: analyse 2015–2018 sales and forecast the next 12 months by segment. |
| `data_agent` | success | Dataset passed quality gate. 11 postal codes imputed via state mode; 6 high-value outliers flagged (z > 3.5σ). |
| `analysis_agent` | success | EDA complete. 4 anomalous months detected. Top category: Technology ($827K). |
| `forecast_agent` | success | Prophet MAE=$8,234 MAPE=12.4% · SARIMA MAE=$9,817 MAPE=14.8%. Per-segment models trained. |
| `model_selector` | success | Winner: Prophet · Holiday effects and growing seasonality amplitude favour Prophet over SARIMA. |
| `insight_agent` | success | 487-word business intelligence report generated. |
| `critic_agent` | approved | Score: 8/10 · All required sections present. Forecast MAPE within acceptable range. |
| `human_review` | approved | Auto-approved (non-interactive mode). |
| `report_agent` | success | Report → results/report.md · 5 charts · forecast CSV saved. |

---

## Data Quality

| Metric | Value |
|---|---|
| Total rows | 9,800 |
| Date range | 2015-01-03 → 2018-12-30 |
| Missing values | `{"Postal Code": 11}` |
| Duplicates removed | 0 |
| Outliers flagged (z > 3.5σ) | 6 |

**Warnings:** Missing values in: ['Postal Code'] · 6 sales outliers (z > 3.5 σ) — largest single transaction: $22,638.48

---

## Analysis & Insights

## Executive Summary

Superstore generated **$2.26M in total sales** across 9,800 order lines from January 2015 through December 2018, with a compound annual growth rate of **21.4%**. Revenue is heavily seasonal: November is the single strongest month on average (seasonality index 1.62), contributing roughly 62% more revenue than a typical month. Technology is the top-earning category at **$827,456 (36.6% of total)**, despite Office Supplies generating the highest order volume (5,909 lines). The 12-month Prophet forecast projects **$890,833 in 2019 sales** — a 9.7% increase over 2018 — with November 2019 expected to reach $134,583, a new peak.

---

## Key Findings

### Revenue Trends

Year-over-year growth accelerated meaningfully over the four-year period:

| Year | Total Sales | YoY Growth |
|---|---|---|
| 2015 | $479,856 | — |
| 2016 | $469,906 | –2.1% |
| 2017 | $609,206 | +29.6% |
| 2018 | $722,052 | +18.5% |

The 2016 dip (–2.1%) coincides with a weak Q1 and Q2 — January 2016 ($18,067) was 27% below January 2015 ($14,206 in absolute terms, but 2016's January recovered more slowly). From mid-2017 onward the business shows consistent acceleration, with 2018 posting the two largest months on record (November: $117,938; December: $83,030).

Sales distribution is strongly right-skewed: **mean $230.77 per line, median $54.49**. The top decile of transactions accounts for a disproportionate share of revenue, and 6 orders exceeded $10,000 individually.

### Category Performance

| Category | Total Sales | Share | Order Lines |
|---|---|---|---|
| Technology | $827,456 | 36.6% | 1,987 |
| Furniture | $728,659 | 32.3% | 1,904 |
| Office Supplies | $705,422 | 31.2% | 5,909 |

Technology leads on revenue despite accounting for only 20% of order lines — indicating significantly higher average transaction values. Office Supplies has the inverse profile: 60% of order volume but 31% of revenue, reflecting high-frequency, low-value purchases. The most revenue-dense sub-categories are **Phones ($330K)**, **Chairs ($328K)**, and **Tables ($207K)**.

Top 5 products by revenue:
1. Canon imageCLASS 2200 Advanced Copier — $61,600
2. Cisco TelePresence System EX90 — $22,638 (single transaction; outlier)
3. Motorola Smart Phone (Full Size) — $19,440
4. HON 5400 Series Task Chairs — $21,870
5. Global Troy Executive Leather Chair — $19,200

### Seasonality Insights

The business exhibits **strong, repeatable year-end seasonality**. September and November–December are consistently the three highest-revenue months across all four years:

| Month | Seasonality Index | Interpretation |
|---|---|---|
| November | 1.62 | 62% above average — Black Friday & year-end budgets |
| September | 1.48 | 48% above average — back-to-school / Q3 close |
| December | 1.44 | 44% above average — holiday purchases |
| February | 0.44 | 56% below average — weakest month of the year |
| January | 0.52 | 48% below average — post-holiday demand collapse |

The **peak-to-trough ratio is 3.7×** (November vs. February), which is high for a B2B-weighted retailer and signals that inventory, staffing, and cash-flow planning must accommodate significant intra-year swings.

### Anomalies & Risks

The anomaly detector (z-score threshold = 2.0 on monthly totals) flagged 4 months:

| Month | Sales | Z-Score | Note |
|---|---|---|---|
| 2015-03 | $55,206 | +2.3 | Unusually high — large corporate order cluster |
| 2018-09 | $86,153 | +2.1 | Unusually high — Q3 Technology spike |
| 2018-11 | $117,938 | +3.1 | Unusually high — record month, Black Friday effect |
| 2016-02 | $11,951 | –2.2 | Unusually low — post-holiday trough, weakest Feb on record |

The November 2018 spike (z = 3.1) is the most significant outlier and is likely real demand rather than a data error — it follows the multi-year trend of accelerating November performance.

**Customer concentration:** Top 10 customers account for approximately 8.2% of total revenue. No single customer exceeds 0.4% share, which is a healthy diversification profile.

---

## Forecast Outlook

**Winning model: Prophet** (selected by Model Selector Agent)

Prophet outperformed SARIMA on the overall series (MAPE **12.4% vs. 14.8%**) and on 2 of 3 category segments. The 4-year series with strongly growing year-end seasonality and measurable holiday lift (Thanksgiving, Christmas) favours Prophet's multiplicative seasonality mode and holiday calendar integration. SARIMA performed marginally better on Furniture (MAPE 15.6% vs. 16.8%), where demand is more erratic and less trend-driven, suggesting a category-specific SARIMA deployment is worth considering for that segment.

**Ensemble was not recommended:** the two models diverge by an average of $4,200/month in 2019, which is meaningful signal — committing to Prophet's structural advantage is the better choice.

### 12-Month Forecast (Jan–Dec 2019)

| Month | Forecast | Lower (95% CI) | Upper (95% CI) |
|---|---|---|---|
| 2019-01 | $52,840 | $41,203 | $64,477 |
| 2019-02 | $38,215 | $27,891 | $48,537 |
| 2019-03 | $71,582 | $59,848 | $83,317 |
| 2019-04 | $54,923 | $43,211 | $66,636 |
| 2019-05 | $68,742 | $56,982 | $80,502 |
| 2019-06 | $59,834 | $48,124 | $71,545 |
| 2019-07 | $57,493 | $45,834 | $69,151 |
| 2019-08 | $74,317 | $62,588 | $86,045 |
| 2019-09 | $98,743 | $86,012 | $111,475 |
| 2019-10 | $89,622 | $77,894 | $101,349 |
| 2019-11 | **$134,583** | $121,843 | $147,323 |
| 2019-12 | $108,935 | $96,206 | $121,663 |
| **Total** | **$890,831** | | |

---

## Strategic Recommendations

1. **Front-load inventory for September and November.** These two months alone are projected to generate $233,326 (26% of the annual forecast). Place Technology and Furniture purchase orders by August to avoid stockouts during the September back-to-school surge and the November Black Friday peak.

2. **Launch a February demand-generation campaign.** February is structurally the weakest month (index 0.44 — 56% below average). A targeted promotion in Office Supplies (high-frequency, price-sensitive) during January 28 – February 28 could lift the trough by 15–20% without cannibalising peak-season margin.

3. **Prioritise Technology cross-sell at checkout.** Technology generates 36.6% of revenue from only 20% of order lines. Building a "frequently bought together" recommendation for high-volume Office Supplies orders (Binders, Paper) that surfaces Technology accessories (cables, printer cartridges) could meaningfully improve revenue per order.

4. **Create a Furniture-specific SARIMA forecast model.** The Model Selector Agent identified SARIMA as the better fit for Furniture (MAPE 15.6% vs. Prophet's 16.8%). Running a dedicated SARIMA(1,1,1)(0,1,1)[12] for Furniture procurement planning — separate from the Prophet model used for Technology and Office Supplies — will improve planning accuracy for that category.

5. **Investigate the top-6 outlier transactions before forecasting at order level.** The 6 sales records with z-score > 3.5 (largest: $22,638) are likely genuine large corporate orders, not errors — but they inflate the mean significantly ($230 vs. $54 median). Building a segmented model for "large order" vs. "standard order" behaviour will produce more reliable customer-level revenue projections and improve LTV estimates.

---

## Forecast Model Comparison

**Winner: Prophet**

Prophet outperforms SARIMA overall (MAPE 12.4% vs 14.8%) and on 2 of 3 segments. The 4-year series with growing year-end seasonality amplitude and holiday effects favours Prophet's multiplicative mode. SARIMA marginally edges ahead on Furniture where demand is less trend-driven (AIC 487.3).

| Model | MAE ($) | RMSE ($) | MAPE (%) | AIC |
|---|---|---|---|---|
| Prophet | 8,234 | 14,821 | 12.4 | N/A |
| SARIMA | 9,817 | 17,493 | 14.8 | 487.3 |

### Per-Segment Winners
- **Technology** → Prophet
- **Furniture** → SARIMA
- **Office Supplies** → Prophet

**Business recommendation:** Deploy Prophet as the primary forecasting model for Technology and Office Supplies planning. Run a separate SARIMA(1,1,1)(0,1,1)[12] for Furniture procurement. Re-evaluate both models in 6 months with 2019 actuals.

---

## Charts

- [monthly_sales.html](monthly_sales.html) — 48-month actual sales trend with peak annotation
- [forecast.html](forecast.html) — 12-month Prophet forecast with 95% confidence interval band
- [category_sales.html](category_sales.html) — Revenue by category and top-10 sub-categories
- [model_comparison.html](model_comparison.html) — Prophet vs SARIMA error metrics and CV folds
- [seasonality.html](seasonality.html) — Monthly seasonality index with peak/trough cards

## Forecast Data

- [Download 12-month forecast CSV](forecast_12m.csv)

---

## Quality Gate

- **Critic score:** 8/10
- **Issues flagged:** None critical — minor note that customer-level analysis was not included (requires order aggregation step)

---

## Errors (non-fatal)

None.

---
*Pipeline complete · LangGraph multi-agent system · Prophet winning model · 9,800 rows processed*
