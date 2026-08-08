# Part 2 — Statistical EDA, Hypothesis Testing & Visualization

## Dataset

Reused Part 1's cleaned output: `customers_orders_cleaned.csv` (16,282 rows, from the
Northwind Customers⟕Orders join, already deduplicated and imputed). Meets the ≥500
rows / ≥6 columns / date-column requirement independently of Part 1.

## How to run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python eda_analysis.py
```

Produces four `.png` charts in this folder and prints every statistic below to the
console.

## Task 1 — Initial inspection

`df.info()` and `df.describe(include='all')` are printed in full by the script (see
console output). 16,282 rows, 12 columns after Part 1's cleaning plus the
`DaysToShip` derived column.

## Task 2 — NumPy fundamentals

`Freight` was converted to a NumPy ndarray (`dtype=float64`, shape `(16282,)`).

- **Vectorized operation:** a 7% surcharge was applied to every value in one line
  (`freight_arr * 1.07`), no loop. First value: 107.75 → 115.29.
- **Two-condition boolean filter:** `(freight_arr >= 100) & (freight_arr <= 300)`
  found **6,355 orders (39.0%)** with Freight between 100 and 300.

## Task 3 — Descriptive statistics (NumPy)

| Column | mean | median | std | var | 90th percentile |
|---|---|---|---|---|---|
| Freight | 248.59 | 245.25 | 148.97 | 22193.33 | 454.00 |
| DaysToShip | 7.30 | 5.00 | 6.78 | 45.94 | 18.00 |

## Task 4 — Feature engineering

`FreightPerDay = Freight / (DaysToShip + 1)` — a "shipping cost per day of transit"
efficiency measure. The `+1` avoids division-by-zero for same-day-shipped orders.

## Task 5 — Grouped analysis

- **Pivot 1** (`index='Country', values='Freight', aggfunc='mean'`): Norway has the
  highest mean freight cost (267.72), followed by Denmark (264.52).
- **Pivot 2** (`index='ShipCountry', values='DaysToShip', aggfunc='median'`): several
  countries (Belgium, Canada, Portugal, Sweden, Denmark, Switzerland, Poland, USA)
  tie at a 6-day median shipping time.
- **Multi-agg groupby:** `df.groupby('Country').agg({'Freight': ['mean','sum'],
  'DaysToShip': ['mean','max']})` — e.g. USA: mean Freight 248.60, total Freight
  spend $566,798.25, mean DaysToShip 7.52, max DaysToShip 29.

## Task 6 — Bucket segmentation

`bucket_freight()` maps `Freight` into three tiers: **Low** (<100), **Medium**
(100–300), **High** (>300), applied via `.apply()` into a new `FreightTier` column.
Result: High=6,445, Medium=6,355, Low=3,482.

## Task 7 — Correlation analysis

Pearson correlation matrix (`df.corr()`) across all four numeric columns (`OrderID`,
`Freight`, `DaysToShip`, `FreightPerDay`):

|  | OrderID | Freight | DaysToShip | FreightPerDay |
|---|---|---|---|---|
| OrderID | 1.00 | 0.14 | 0.01 | 0.04 |
| Freight | 0.14 | 1.00 | 0.00 | 0.42 |
| DaysToShip | 0.01 | 0.00 | 1.00 | -0.55 |
| FreightPerDay | 0.04 | 0.42 | -0.55 | 1.00 |

- **Highest absolute correlation pair (excluding diagonal):** `DaysToShip` &
  `FreightPerDay`, r = **-0.548**. This is expected — `FreightPerDay` is
  *mathematically derived* from `DaysToShip` (it's in the denominator), so a strong
  relationship is inherent to the formula, not a new discovery.
- **Lowest absolute correlation pair:** `DaysToShip` & `Freight`, r = **0.0036**
  (effectively zero) — how quickly an order ships has no relationship with how much
  it costs to ship.
- **Ties:** no two pairs had equal absolute correlation in this run, so no tie-break
  was needed.
- **NaN values:** none of the four numeric columns had zero/near-zero variance, so
  no correlation came back undefined. The script explicitly checks for and would
  report any NaN pairs (`nan_pairs` — printed as empty in this run).

## Task 8-9 — Hypothesis test

**Business claim:** average Freight cost differs between orders from German
customers and orders from French customers (these are the two largest non-USA
customer countries in the dataset, both with large, comparable sample sizes).

- H0: mean(Freight | Germany) = mean(Freight | France)
- H1: mean(Freight | Germany) ≠ mean(Freight | France)
- α = 0.05
- **Test used:** two-sample Welch's t-test (`scipy.stats.ttest_ind(..., equal_var=False)`)
  — Welch's variant was chosen over the standard pooled-variance t-test because it
  doesn't assume the two groups have equal variance, which is a safer default.
- **Assumption check:** approximate normality was checked via skewness — Germany's
  Freight skew is 0.10 and France's is 0.03, both close to 0 (a perfectly symmetric/
  normal distribution has skew 0), so the normality assumption is reasonably met.
  Both groups have large samples (n=1,895 and n=1,909), which also makes the t-test
  robust to any minor residual skew via the Central Limit Theorem.
- **Result:** Germany mean = 244.14, France mean = 251.68. t = -1.5578, p = 0.1194.
- **Decision:** p (0.1194) ≥ α (0.05) → **fail to reject H0**. There is not enough
  statistical evidence in this sample to conclude that average Freight cost genuinely
  differs between German and French orders — the ~$7.54 gap seen in the raw means is
  plausibly due to random sampling variation, not a real underlying difference.

## Task 10 — Visualizations

- `chart_a_correlation_heatmap.png` — annotated Pearson correlation heatmap.
- `chart_b_scatter_freight_vs_days.png` — Freight vs. DaysToShip, coloured by
  Country (top 5 by order count).
- `chart_c_bar_mean_freight_by_country.png` — mean Freight by country, top 10.
- `chart_d_distribution_freight.png` — histogram + KDE of Freight cost.

All four have titles and labelled axes.

## Task 11 — Insight → Recommendation

1. **Insight:** `DaysToShip` and `Freight` are essentially uncorrelated (r = 0.0036)
   — paying more for freight does not get an order shipped faster in this data.
   **Recommendation:** stop treating "expedite by upgrading freight" as an assumed
   lever; if faster shipping is the goal, look at fulfillment/processing changes
   instead of freight spend, since freight cost isn't buying speed here.

2. **Insight:** the Welch's t-test found no statistically significant difference in
   average Freight cost between Germany (244.14) and France (251.68) (p = 0.1194).
   **Recommendation:** don't set different freight-pricing policies for these two
   markets based on a perceived cost gap — the gap isn't statistically distinguishable
   from noise, so a uniform policy is defensible and simpler to operate.

3. **Insight:** 39.6% of orders (6,445 of 16,282) fall in the "High" Freight tier
   (>$300), representing the largest single tier by order count.
   **Recommendation:** since High-tier orders are the modal case rather than a rare
   outlier group, prioritize negotiating better shipping-carrier rates specifically
   for this tier — it has the largest total dollar impact of the three tiers.

4. **Insight:** Norway has the highest mean Freight cost per order (267.72) among all
   countries in the pivot table, despite not being a top-5 country by order volume.
   **Recommendation:** investigate Norway's shipping route/carrier setup specifically
   — a consistently elevated per-order freight cost in a lower-volume market often
   signals an inefficient shipping arrangement worth renegotiating.
