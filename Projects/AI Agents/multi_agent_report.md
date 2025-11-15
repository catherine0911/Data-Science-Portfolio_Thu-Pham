# Multi-Agent Analysis Report

## Executive Summary
Here’s a concise, actionable summary of the dataset and the main analysis findings.

Dataset overview
- Rows: 9,800 order-line records.
- Orders: 4,922 unique Order IDs (many orders contain multiple line items).
- Customers: 793 unique customers (top: WB-21850 / William Brown — 35 lines).
- Products: 1,849 unique product names, 1,861 unique product IDs.
- Geography: all records are United States; 49 states represented. Top state: California (1,946 rows). Top city: New York City (891 rows).
- Dates: Orders span Jan 2015 → Dec 2018 (48 months).

Sales distribution (numeric)
- Mean sale per line: $230.77; median: $54.49 → strongly right-skewed.
- Std dev: $626.65; min: $0.444; 25%: $17.25; 75%: $210.61; max: $22,638.48 → presence of high-value outliers.
- Implication: most line items are low-value, a small number of large transactions drive much revenue.

Category / product / segment highlights
- Top categories by total sales:
  1. Technology — $827,455.87
  2. Furniture — $728,658.58
  3. Office Supplies — $705,422.33
- Most frequent category/sub-category/product:
  - Category (by count): Office Supplies (5,909 lines)
  - Top sub-category by frequency: Binders (1,492 lines)
  - Top product by frequency: “Staple envelope” (47 occurrences)
- Segment: Consumer is largest by count (5,101 records).
- Ship Mode: Standard Class dominates (5,859 rows).

Time-series (monthly trend)
- Monthly sales series (Jan 2015 → Dec 2018) shows clear seasonality with recurring high months in Nov–Dec.
- The largest single-month revenue occurs Nov 2018: $117,938.16, followed by strong December months (e.g., Dec 2017 ≈ $95,739).
- General pattern: repeated year-end peaks and several other intermittent spikes; overall activity increases toward the end of the period (2017–2018 show larger peaks than 2015–2016).

Correlation and numeric relationships
- Correlation between Sales and available numeric fields is negligible:
  - Sales vs Postal Code: r ≈ -0.024
  - Sales vs Row ID: r ≈ 0.001
- Interpretation: no simple linear relationship between sales and those numeric fields — use categorical features and engineered features (customer, product, time) for predictive models.

Data quality notes
- Postal Code: count = 9,789 (11 missing postal codes).
- Many duplicate Order IDs (expected for multi-line orders); ensure aggregation level is clear when analyzing revenue per order.
- Some summary fields (Row ID “unique” shown as NaN) are from descriptive output formatting — but primary counts are clear.

Key takeaways & recommended next steps
1. Leverage seasonality:
   - Plan promotions, inventory, and staffing for Nov–Dec (largest, repeatable revenue spikes).
   - Consider Black Friday / holiday campaigns focused on Technology and high-margin Furniture items.
2. Focus sales efforts:
   - Technology brings the most revenue; invest in marketing and cross-sell/upsell in this category.
   - Office Supplies have high frequency but similar total sales — evaluate margins and bundling opportunities (high volume but lower-ticket).
3. Customer / geography targeting:
   - Prioritize outreach and retention for top customers and high-volume states (California, New York City).
4. Inventory & shipping:
   - Standard Class is the default ship mode; during peak months confirm capacity and consider expedited options for higher-margin orders.
5. Data cleanup & deeper analysis:
   - Fill/verify missing postal codes and validate outlier transactions (very large sales).
   - Aggregate to order and customer level to analyze lifetime value, repeat purchase behavior, and average order value.
   - Build time-series forecasts by category to support inventory and procurement decisions.
6. Modeling guidance:
   - Because numeric correlations are weak, include categorical encodings (product, category, customer, state), date features (month, holiday flags), and historical rolling aggregates for predictive tasks.

If you’d like, I can:
- Produce year-by-year revenue totals and growth rates,
- Show top 10 products/ customers by revenue,
- Build a monthly seasonality plot and a simple forecast for the next 6–12 months,
- Or aggregate at the order or customer level for lifetime value analysis. Which would you prefer next?

## Analysis
### Describe
**Row ID**:
| Metric | Value |
|---|---|
| count | 9800.0 |
| unique | nan |
| top | nan |
| freq | nan |
| mean | 4900.5 |
| std | 2829.1606529145706 |
| min | 1.0 |
| 25% | 2450.75 |
| 50% | 4900.5 |
| 75% | 7350.25 |
| max | 9800.0 |

**Order ID**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 4922 |
| top | CA-2018-100111 |
| freq | 14 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Order Date**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 1230 |
| top | 05/09/2017 |
| freq | 38 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Ship Date**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 1326 |
| top | 26/09/2018 |
| freq | 34 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Ship Mode**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 4 |
| top | Standard Class |
| freq | 5859 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Customer ID**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 793 |
| top | WB-21850 |
| freq | 35 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Customer Name**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 793 |
| top | William Brown |
| freq | 35 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Segment**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 3 |
| top | Consumer |
| freq | 5101 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Country**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 1 |
| top | United States |
| freq | 9800 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**City**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 529 |
| top | New York City |
| freq | 891 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**State**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 49 |
| top | California |
| freq | 1946 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Postal Code**:
| Metric | Value |
|---|---|
| count | 9789.0 |
| unique | nan |
| top | nan |
| freq | nan |
| mean | 55273.322402696904 |
| std | 32041.223412812957 |
| min | 1040.0 |
| 25% | 23223.0 |
| 50% | 58103.0 |
| 75% | 90008.0 |
| max | 99301.0 |

**Region**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 4 |
| top | West |
| freq | 3140 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Product ID**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 1861 |
| top | OFF-PA-10001970 |
| freq | 19 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Category**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 3 |
| top | Office Supplies |
| freq | 5909 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Sub-Category**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 17 |
| top | Binders |
| freq | 1492 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Product Name**:
| Metric | Value |
|---|---|
| count | 9800 |
| unique | 1849 |
| top | Staple envelope |
| freq | 47 |
| mean | nan |
| std | nan |
| min | nan |
| 25% | nan |
| 50% | nan |
| 75% | nan |
| max | nan |

**Sales**:
| Metric | Value |
|---|---|
| count | 9800.0 |
| unique | nan |
| top | nan |
| freq | nan |
| mean | 230.76905945918367 |
| std | 626.6518748388042 |
| min | 0.444 |
| 25% | 17.248 |
| 50% | 54.489999999999995 |
| 75% | 210.60500000000002 |
| max | 22638.48 |

### Monthly Trend
- Sales: 48 months of data

### Top by Category
| Category | Sales |
|---|---|
| Technology | 827455.87 |
| Furniture | 728658.58 |
| Office Supplies | 705422.33 |

### Correlation Matrix
- **Row ID**: Row ID: 1.000, Postal Code: 0.014, Sales: 0.001
- **Postal Code**: Row ID: 0.014, Postal Code: 1.000, Sales: -0.024
- **Sales**: Row ID: 0.001, Postal Code: -0.024, Sales: 1.000

## Visuals
- See `results/` folder for charts (monthly trend, sales distribution, top categories)
