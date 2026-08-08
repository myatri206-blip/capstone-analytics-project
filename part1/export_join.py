"""
Task 6 setup — exports the Task 4 LEFT JOIN (Customers LEFT JOIN Orders)
to a CSV file, which the cleaning script then loads.

We deliberately include ShippedDate and OrderDate (needed later to derive
a 'DaysToShip' continuous measure) and Freight (a continuous numeric
measure) so the cleaning/outlier steps in Task 7-8 have real material
to work with.
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect("northwind.db")
conn.execute("PRAGMA foreign_keys = ON;")

query = """
SELECT
    c.CustomerID,
    c.CompanyName,
    c.Country,
    c.Region,
    o.OrderID,
    o.OrderDate,
    o.ShippedDate,
    o.Freight,
    o.ShipCountry
FROM Customers c
LEFT JOIN Orders o ON c.CustomerID = o.CustomerID;
"""

df = pd.read_sql_query(query, conn)
conn.close()

df.to_csv("customers_orders_joined.csv", index=False)
print(f"Exported {len(df)} rows, {len(df.columns)} columns to customers_orders_joined.csv")
print(df.head())
