# Part 1 — Data Foundations: SQL Extraction, Cleaning & Outlier Audit

## Dataset

[Northwind](https://github.com/jpwhite3/northwind-SQLite3) (MIT License), SQLite format,
downloaded directly from the prebuilt `.db` file. Two tables are used:

- **Customers** — 93 rows, 11 columns. `CustomerID` is the **PRIMARY KEY**.
- **Orders** — 16,282 rows, 14 columns, including `OrderDate` (date column). `CustomerID`
  is a **FOREIGN KEY** referencing `Customers.CustomerID`.

Schema (abridged, see `queries.sql` header comment for the full column list):

```sql
CREATE TABLE Customers (
    CustomerID TEXT PRIMARY KEY,
    CompanyName TEXT, ... , Country TEXT
);

CREATE TABLE Orders (
    OrderID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID TEXT,
    OrderDate DATETIME, ShippedDate DATETIME, Freight NUMERIC, ...,
    FOREIGN KEY (CustomerID) REFERENCES Customers (CustomerID)
);
```

This satisfies the brief's requirement of ≥500 rows, ≥6 columns, a date column, and a
real key relationship spanning two tables.

## How to run

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python fk_enforcement_test.py  # Task 1: proves FK enforcement is active
sqlite3 northwind.db < queries.sql   # Task 2-5: runs all SQL queries
python export_join.py          # Task 6 setup: exports join to CSV
python clean_and_audit.py      # Task 6-7: cleaning + outlier audit
```

## Task 1 — Foreign-key enforcement

SQLite does **not** enforce FK constraints by default. `fk_enforcement_test.py` runs
`PRAGMA foreign_keys = ON;`, confirms the pragma value is `1`, then attempts to insert
an `Orders` row with `CustomerID = 'ZZZZZ'` (an ID that does not exist in `Customers`).
The result:

```
foreign_keys pragma value: 1 (1 = ON, 0 = OFF)
INSERT REJECTED as expected. SQLite raised: FOREIGN KEY constraint failed
```

This proves enforcement is genuinely active, not just declared in the DDL.

## Task 2 — Six SQL query techniques

All six live in `queries.sql`: `IN`, `NOT IN`, `BETWEEN` (on `OrderDate`, since this
Northwind fork's dates run 2012–2023 rather than the classic 1997, `2018-01-01` to
`2018-12-31` was used to get non-empty results), a two-column `ORDER BY` (CustomerID
ascending, OrderDate descending), a `NOT EXISTS` subquery for customers with no
orders, and a `LIKE '%...%'` text search.

**Note on the `NOT EXISTS` result:** it returns 0 rows. Cross-checked against Task 5a
below — all 93 customers have at least one order in this dataset, so "customers with no
orders" is legitimately empty here. This is a real finding, not a query bug.

## Task 3 — GROUP BY + HAVING

```sql
SELECT CustomerID, COUNT(*) AS order_count, AVG(Freight) AS avg_freight
FROM Orders
GROUP BY CustomerID
HAVING COUNT(*) > 5;
```

Two aggregates (`COUNT`, `AVG`), filtered with `HAVING` (not `WHERE`, since the filter
is on an aggregated value that doesn't exist until after grouping). Returns 93 rows —
every customer in this dataset has more than 5 orders.

## Task 4 — INNER JOIN and LEFT JOIN

`Customers` is placed on the **left** in both joins. Justification: the analytical
question driving both queries is "for each customer, what orders do they have?" — the
customer is the entity we're reporting on, so we need the full customer set retained.

- **INNER JOIN** — keeps only customers that have a matching order. Returned 16,282
  rows (matches the Orders row count, confirming every order has a valid customer).
- **LEFT JOIN** — keeps *every* customer, with `NULL` order columns for any customer
  with zero orders. Also returned 16,282 rows here — identical to the INNER JOIN
  result, which independently confirms the Task 2e finding that no customer is
  orderless in this dataset.

## Task 5 — Referential integrity validation

| Check | Query | Result |
|---|---|---|
| 5a. COUNT(DISTINCT) sanity check | distinct `CustomerID`s in Orders vs. total Customers | 93 vs. 93 — every customer appears in Orders at least once |
| 5b. Grouped child-count | `GROUP BY CustomerID, COUNT(*)` on Orders | max is 210 orders for one customer (`BSBEV`) — confirms this is a **1:many** relationship, not 1:1 |
| 5c. Orphan check | Orders whose `CustomerID` has no matching Customers row | **0 rows** — no orphans found |

**Conclusion:** Customers→Orders is a clean **1:many** relationship with zero orphaned
rows. The relationship is fully intact.

## Task 6 — Cleaning the exported CSV

The Task 4 LEFT JOIN (Customers LEFT JOIN Orders, 9 columns, 16,282 rows) was exported
to `customers_orders_joined.csv` via `export_join.py`.

**6a. Missing values (before imputation):**

| Column | Missing count | Missing % |
|---|---|---|
| Country | 335 | 2.06% |
| Region | 335 | 2.06% |
| ShippedDate | 21 | 0.13% |
| all other columns | 0 | 0% |

**6b. Imputation strategy:** No numeric column had missing values in this export, so
only the categorical rule applied: `Country`, `Region`, and `ShippedDate` were filled
with the literal string `"unknown"`. (Had a numeric column needed imputation, **median**
would have been used over mean, because `Freight` — the one continuous numeric measure
in this data — is right-skewed by a small number of large-value shipments; the mean is
pulled upward by those, while the median stays representative of a "typical" order.)
After imputation, `df.isnull().sum()` is 0 for every column (verified by an `assert` in
the script).

**6c. Duplicates:** 16,282 rows before `drop_duplicates()`, 16,282 rows after — **0
duplicates found**.

## Task 7 — Outlier audit (IQR vs. Z-score)

**Filtering rule for "continuous numeric measure":** exclude ID/key columns
(`OrderID` — its magnitude is meaningless, it's an identifier), exclude binary/flag
columns (none present), and exclude near-zero-variance columns (none present). A
derived column, `DaysToShip` (`ShippedDate − OrderDate`, in days), was computed because
it is a genuine continuous business measure even though it isn't a raw CSV column.

**Surviving measures: `Freight`, `DaysToShip`.**

| Measure | Q1 | Q3 | IQR | Fences | IQR outliers | Mean | Std | Z-score outliers (|Z|>3) |
|---|---|---|---|---|---|---|---|---|
| Freight | 117.25 | 377.25 | 260.00 | [-272.75, 767.25] | **0** | 248.59 | 148.98 | **0** |
| DaysToShip | 2.00 | 11.00 | 9.00 | [-11.50, 24.50] | **278** | 7.30 | 6.78 | **45** |

**Do the methods agree?**
- On **Freight**, yes — both flag 0 outliers. No order's freight cost is extreme enough
  to trip either fence.
- On **DaysToShip**, no — IQR flags 278 rows, Z-score flags only 45. This is expected:
  `DaysToShip` is right-skewed (most orders ship within ~2–11 days, but a long tail
  takes much longer). The IQR method is based on quartiles and is robust to skew, so it
  correctly flags the whole long tail. The Z-score method uses the mean and standard
  deviation, both of which are themselves inflated by that same long tail — this widens
  the ±3σ band and makes the test under-sensitive, so it misses many of the same points
  IQR catches. This is a textbook example of why IQR is generally preferred for skewed
  distributions.
