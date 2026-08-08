"""
Part 2 — Statistical EDA, Hypothesis Testing & Visualization
Dataset: customers_orders_cleaned.csv (Part 1 output, Northwind-derived)

Run: python eda_analysis.py
Produces 4 PNG charts in this folder and prints everything the README quotes.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it can run with no display, e.g. on a server
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# ================================================================
# Task 1: Initial inspection
# ================================================================
df = pd.read_csv("customers_orders_cleaned.csv")

print("=" * 70)
print("TASK 1 — df.info()")
print("=" * 70)
df.info()

print("\n" + "=" * 70)
print("TASK 1 — df.describe(include='all')")
print("=" * 70)
print(df.describe(include="all"))

# ================================================================
# Task 2: NumPy fundamentals
# ================================================================
print("\n" + "=" * 70)
print("TASK 2 — NumPy vectorized operation and boolean filtering")
print("=" * 70)

freight_arr = df["Freight"].to_numpy()
print(f"Freight as ndarray, dtype={freight_arr.dtype}, shape={freight_arr.shape}")

# Vectorized arithmetic: apply a 7% fuel-surcharge adjustment to every
# value in one line, no Python loop.
freight_with_surcharge = freight_arr * 1.07
print(f"First 5 raw Freight values:        {freight_arr[:5]}")
print(f"First 5 with 7% surcharge applied: {freight_with_surcharge[:5]}")

# Boolean-indexed filtering combining two conditions with &
mid_range_mask = (freight_arr >= 100) & (freight_arr <= 300)
mid_range_values = freight_arr[mid_range_mask]
print(f"\nOrders with Freight between 100 and 300 (inclusive): {len(mid_range_values)} "
      f"out of {len(freight_arr)} ({len(mid_range_values)/len(freight_arr)*100:.1f}%)")

# ================================================================
# Task 3: Descriptive statistics via NumPy (two numeric columns)
# ================================================================
print("\n" + "=" * 70)
print("TASK 3 — Descriptive statistics (NumPy)")
print("=" * 70)

days_arr = df["DaysToShip"].dropna().to_numpy()

for name, arr in [("Freight", freight_arr), ("DaysToShip", days_arr)]:
    print(f"\n--- {name} ---")
    print(f"  mean   = {np.mean(arr):.3f}")
    print(f"  median = {np.median(arr):.3f}")
    print(f"  std    = {np.std(arr):.3f}")
    print(f"  var    = {np.var(arr):.3f}")
    print(f"  90th percentile = {np.percentile(arr, 90):.3f}")

# ================================================================
# Task 4: Feature engineering
# FreightPerDay = Freight / (DaysToShip + 1)
# A business-meaningful "shipping cost efficiency" measure: freight
# dollars spent per day of transit time. The +1 avoids division by
# zero for same-day-shipped orders.
# ================================================================
print("\n" + "=" * 70)
print("TASK 4 — Feature engineering: FreightPerDay")
print("=" * 70)
df["FreightPerDay"] = df["Freight"] / (df["DaysToShip"].fillna(0) + 1)
print(df[["OrderID", "Freight", "DaysToShip", "FreightPerDay"]].head())

# ================================================================
# Task 5: Grouped analysis — two pivot tables + one multi-agg groupby
# ================================================================
print("\n" + "=" * 70)
print("TASK 5 — Pivot tables")
print("=" * 70)

pivot1 = df.pivot_table(index="Country", values="Freight", aggfunc="mean")
print("\nPivot 1: mean Freight by Country")
print(pivot1.sort_values("Freight", ascending=False).head(10))

pivot2 = df.pivot_table(index="ShipCountry", values="DaysToShip", aggfunc="median")
print("\nPivot 2: median DaysToShip by ShipCountry")
print(pivot2.sort_values("DaysToShip", ascending=False).head(10))

print("\nMulti-aggregation groupby().agg() — two functions x two columns, by Country")
multi_agg = df.groupby("Country").agg({
    "Freight": ["mean", "sum"],
    "DaysToShip": ["mean", "max"],
})
print(multi_agg.sort_values(("Freight", "sum"), ascending=False).head(10))

# ================================================================
# Task 6: Bucket segmentation via .apply()
# ================================================================
print("\n" + "=" * 70)
print("TASK 6 — Bucket segmentation")
print("=" * 70)

def bucket_freight(cost):
    """Maps a Freight value into one of three labelled cost tiers."""
    if cost < 100:
        return "Low"
    elif cost <= 300:
        return "Medium"
    else:
        return "High"

df["FreightTier"] = df["Freight"].apply(bucket_freight)
print(df[["OrderID", "Freight", "FreightTier"]].head())
print("\nFreightTier value counts:")
print(df["FreightTier"].value_counts())

# ================================================================
# Task 7: Correlation analysis
# ================================================================
print("\n" + "=" * 70)
print("TASK 7 — Correlation analysis")
print("=" * 70)

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print(corr_matrix)

# Find highest and lowest absolute correlation pair, excluding the diagonal
corr_unstacked = corr_matrix.where(
    ~np.eye(len(corr_matrix), dtype=bool)
).unstack().dropna()
# Each pair appears twice (A,B) and (B,A) - dedupe by sorting the pair
seen = set()
pairs = []
for (a, b), val in corr_unstacked.items():
    key = tuple(sorted([a, b]))
    if key not in seen:
        seen.add(key)
        pairs.append((key[0], key[1], val))

pairs_df = pd.DataFrame(pairs, columns=["col_a", "col_b", "corr"])
pairs_df["abs_corr"] = pairs_df["corr"].abs()
nan_pairs = pairs_df[pairs_df["corr"].isna()]
valid_pairs = pairs_df.dropna(subset=["corr"]).sort_values("abs_corr", ascending=False)

print(f"\nPairs with NaN correlation (undefined, e.g. zero variance): {len(nan_pairs)}")
if len(nan_pairs) > 0:
    print(nan_pairs)

print(f"\nHighest absolute correlation pair:\n{valid_pairs.iloc[0]}")
print(f"\nLowest absolute correlation pair:\n{valid_pairs.iloc[-1]}")

# ================================================================
# Task 8-9: Hypothesis test
# Claim: "The average Freight cost for orders from customers in
# Germany differs from the average Freight cost for orders from
# customers in France."
# H0: mean(Freight | Germany) == mean(Freight | France)
# H1: mean(Freight | Germany) != mean(Freight | France)
# alpha = 0.05, two-sample t-test (Welch's, unequal variance assumed)
# ================================================================
print("\n" + "=" * 70)
print("TASK 8-9 — Hypothesis test")
print("=" * 70)

germany_freight = df.loc[df["Country"] == "Germany", "Freight"]
france_freight = df.loc[df["Country"] == "France", "Freight"]

print(f"Germany: n={len(germany_freight)}, mean={germany_freight.mean():.2f}, "
      f"skew={germany_freight.skew():.2f}")
print(f"France:  n={len(france_freight)}, mean={france_freight.mean():.2f}, "
      f"skew={france_freight.skew():.2f}")

t_stat, p_value = stats.ttest_ind(germany_freight, france_freight, equal_var=False)
alpha = 0.05
print(f"\nWelch's two-sample t-test: t = {t_stat:.4f}, p = {p_value:.4f}")
if p_value < alpha:
    print(f"p < {alpha} -> REJECT H0: the means are significantly different.")
else:
    print(f"p >= {alpha} -> FAIL TO REJECT H0: no significant difference detected.")

# ================================================================
# Task 10: Visualizations
# ================================================================
print("\n" + "=" * 70)
print("TASK 10 — Generating visualizations")
print("=" * 70)

sns.set_theme(style="whitegrid")

# a. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Columns")
plt.tight_layout()
plt.savefig("chart_a_correlation_heatmap.png", dpi=150)
plt.close()

# b. Scatter plot with hue
top5_countries = df["Country"].value_counts().head(5).index
plot_df = df[df["Country"].isin(top5_countries)]
plt.figure(figsize=(9, 6))
sns.scatterplot(data=plot_df, x="DaysToShip", y="Freight", hue="Country", alpha=0.5)
plt.title("Freight vs. Days to Ship, by Customer Country (Top 5 Countries)")
plt.xlabel("Days to Ship")
plt.ylabel("Freight Cost ($)")
plt.tight_layout()
plt.savefig("chart_b_scatter_freight_vs_days.png", dpi=150)
plt.close()

# c. Bar plot of aggregated value by category
plt.figure(figsize=(9, 6))
bar_data = pivot1.sort_values("Freight", ascending=False).head(10).reset_index()
sns.barplot(data=bar_data, x="Country", y="Freight", color="steelblue")
plt.title("Mean Freight Cost by Country (Top 10)")
plt.xlabel("Country")
plt.ylabel("Mean Freight Cost ($)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("chart_c_bar_mean_freight_by_country.png", dpi=150)
plt.close()

# d. Distribution plot
plt.figure(figsize=(9, 6))
sns.histplot(df["Freight"], bins=40, kde=True)
plt.title("Distribution of Freight Cost")
plt.xlabel("Freight Cost ($)")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("chart_d_distribution_freight.png", dpi=150)
plt.close()

print("Saved: chart_a_correlation_heatmap.png")
print("Saved: chart_b_scatter_freight_vs_days.png")
print("Saved: chart_c_bar_mean_freight_by_country.png")
print("Saved: chart_d_distribution_freight.png")

print("\nDone.")
