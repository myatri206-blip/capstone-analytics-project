"""
Task 1 — Demonstrates that foreign-key enforcement is ACTIVE in SQLite,
not just declared in the schema.

SQLite disables FK enforcement by default. We must run
PRAGMA foreign_keys = ON; on every connection, then prove it is working
by attempting an invalid insert (a CustomerID in Orders that does not
exist in Customers) and showing SQLite rejects it.
"""
import sqlite3

conn = sqlite3.connect("northwind.db")
cur = conn.cursor()

# Step 1: turn enforcement on for this connection
cur.execute("PRAGMA foreign_keys = ON;")

# Step 2: confirm the pragma is actually set
cur.execute("PRAGMA foreign_keys;")
print("foreign_keys pragma value:", cur.fetchone()[0], "(1 = ON, 0 = OFF)")

# Step 3: attempt an invalid insert — 'ZZZZZ' does not exist in Customers
bad_customer_id = "ZZZZZ"
print(f"\nAttempting to insert an Order with CustomerID='{bad_customer_id}' "
      f"(this ID does NOT exist in Customers)...")

try:
    cur.execute(
        """
        INSERT INTO Orders (CustomerID, EmployeeID, OrderDate)
        VALUES (?, 1, '2026-01-01')
        """,
        (bad_customer_id,),
    )
    conn.commit()
    print("INSERT SUCCEEDED — this would mean FK enforcement is NOT active (bad).")
except sqlite3.IntegrityError as e:
    print("INSERT REJECTED as expected. SQLite raised:", e)
    conn.rollback()

conn.close()
