"""
Task 6 — Clean the exported CSV in pandas.
Task 7 — Audit outliers with both IQR and Z-score methods.

Run: python clean_and_audit.py
"""
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Task 6a: Load and report missing values
# ---------------------------------------------------------------
df = pd.read_csv("customers_orders_joined.csv")

print("=" * 60)
print("TASK 6a — Missing values per column (count and %)")
print("=" * 60)
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
print(missing_report)

# ---------------------------------------------------------------
# Task 6b: Imputation
# Numeric columns -> MEDIAN (not mean), because Freight is known to
# be right-skewed with high-value outliers (large freight shipments).
# The mean is pulled upward by those outliers, so the median is a
# more robust "typical value" estimate for imputation. See README.
# Categorical/text columns -> fill with "unknown"
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 6b — Imputation")
print("=" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Filled numeric column '{col}' with median = {median_val}")

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna("unknown")
        print(f"  Filled categorical column '{col}' with 'unknown'")

print("\nRemaining missing values after imputation (should be all 0):")
print(df.isnull().sum())
assert df.isnull().sum().sum() == 0, "Imputation incomplete!"

# ---------------------------------------------------------------
# Task 6c: Duplicate detection and removal
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 6c — Duplicate rows")
print("=" * 60)
rows_before = len(df)
df = df.drop_duplicates()
rows_after = len(df)
print(f"Rows before drop_duplicates(): {rows_before}")
print(f"Rows after  drop_duplicates(): {rows_after}")
print(f"Duplicates removed: {rows_before - rows_after}")

# ---------------------------------------------------------------
# Task 7: Outlier audit
# Filtering rule for "continuous numeric measure":
#   - Exclude ID/key columns (OrderID — it's an identifier, not a
#     measurement; its magnitude is meaningless).
#   - Exclude binary/flag columns (none present here).
#   - Exclude near-zero-variance columns (none present here).
#   - A derived column, DaysToShip (ShippedDate - OrderDate in days),
#     is created because it IS a genuine continuous business measure
#     even though it isn't a raw column in the CSV.
#   -> Surviving continuous measures: Freight, DaysToShip
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 7 — Outlier audit (IQR and Z-score)")
print("=" * 60)

df["OrderDate_parsed"] = pd.to_datetime(df["OrderDate"], errors="coerce")
df["ShippedDate_parsed"] = pd.to_datetime(
    df["ShippedDate"].replace("unknown", np.nan), errors="coerce"
)
df["DaysToShip"] = (df["ShippedDate_parsed"] - df["OrderDate_parsed"]).dt.days
# Rows where ShippedDate was missing/imputed as "unknown" can't have a
# real DaysToShip; drop those from THIS specific measure only (they were
# already handled by the Task 6 imputation for the raw columns).
days_to_ship_valid = df["DaysToShip"].dropna()

continuous_measures = {
    "Freight": df["Freight"],
    "DaysToShip": days_to_ship_valid,
}

results = {}
for name, series in continuous_measures.items():
    series = series.astype(float)

    # --- IQR method ---
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    iqr_outliers = series[(series < lower_fence) | (series > upper_fence)]

    # --- Z-score method ---
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    z_outliers = series[z_scores.abs() > 3]

    results[name] = {
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_fence": lower_fence, "upper_fence": upper_fence,
        "iqr_outlier_count": len(iqr_outliers),
        "mean": mean, "std": std,
        "zscore_outlier_count": len(z_outliers),
    }

    print(f"\n--- {name} ---")
    print(f"  Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}")
    print(f"  Fences: [{lower_fence:.2f}, {upper_fence:.2f}]")
    print(f"  IQR method flags:     {len(iqr_outliers)} rows")
    print(f"  Mean={mean:.2f}  Std={std:.2f}")
    print(f"  Z-score method flags: {len(z_outliers)} rows (|Z| > 3)")

# Save the cleaned CSV for the repo
df.to_csv("customers_orders_cleaned.csv", index=False)
print("\nSaved cleaned data to customers_orders_cleaned.csv")

print("\n" + "=" * 60)
print("Summary table for README")
print("=" * 60)
summary = pd.DataFrame(results).T
print(summary)
