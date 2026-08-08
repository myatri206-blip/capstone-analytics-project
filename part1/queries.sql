-- ============================================================
-- Part 1 — SQL Queries against Northwind (Customers, Orders)
-- Schema reminder:
--   Customers(CustomerID PK, CompanyName, ContactName, ... Country)
--   Orders(OrderID PK, CustomerID FK -> Customers.CustomerID,
--          EmployeeID FK, OrderDate, RequiredDate, ShippedDate,
--          Freight, ShipCountry, ...)
-- Run PRAGMA foreign_keys = ON; on every connection before using
-- this schema (see fk_enforcement_test.py for a live demonstration).
-- ============================================================

-- ---------- Task 2a: WHERE ... IN ----------
SELECT CustomerID, CompanyName, Country
FROM Customers
WHERE Country IN ('Germany', 'France', 'UK');

-- ---------- Task 2b: WHERE ... NOT IN (same column) ----------
SELECT CustomerID, CompanyName, Country
FROM Customers
WHERE Country NOT IN ('Germany', 'France', 'UK');

-- ---------- Task 2c: BETWEEN on a date column ----------
-- (this Northwind fork's OrderDate spans 2012-2023, not the classic 1997)
SELECT OrderID, CustomerID, OrderDate
FROM Orders
WHERE OrderDate BETWEEN '2018-01-01' AND '2018-12-31';

-- ---------- Task 2d: ORDER BY two columns, one ASC one DESC ----------
SELECT OrderID, CustomerID, OrderDate, Freight
FROM Orders
ORDER BY CustomerID ASC, OrderDate DESC;

-- ---------- Task 2e: Subquery — customers with NO matching orders ----------
-- Uses NOT EXISTS (preferred over NOT IN, which breaks silently on NULLs)
SELECT c.CustomerID, c.CompanyName
FROM Customers c
WHERE NOT EXISTS (
    SELECT 1 FROM Orders o WHERE o.CustomerID = c.CustomerID
);

-- ---------- Task 2f: LIKE with % wildcard ----------
SELECT CustomerID, CompanyName
FROM Customers
WHERE CompanyName LIKE '%Market%';

-- ============================================================
-- Task 3: GROUP BY + HAVING with two aggregate functions
-- ============================================================
SELECT
    CustomerID,
    COUNT(*)      AS order_count,
    AVG(Freight)  AS avg_freight
FROM Orders
GROUP BY CustomerID
HAVING COUNT(*) > 5;

-- ============================================================
-- Task 4: INNER JOIN and LEFT JOIN with table aliases
-- Customers is placed on the LEFT in both, because the analysis
-- question we care about is "for each customer, what orders (if
-- any) do they have?" — i.e. we need every customer retained,
-- even ones with zero orders. See README for full justification.
-- ============================================================

-- INNER JOIN: only customers that HAVE at least one order
SELECT c.CustomerID, c.CompanyName, o.OrderID, o.OrderDate
FROM Customers c
INNER JOIN Orders o ON c.CustomerID = o.CustomerID;

-- LEFT JOIN: ALL customers, with NULL order columns for those with none
SELECT c.CustomerID, c.CompanyName, o.OrderID, o.OrderDate
FROM Customers c
LEFT JOIN Orders o ON c.CustomerID = o.CustomerID;

-- ============================================================
-- Task 5: Referential integrity validation (three required checks)
-- ============================================================

-- 5a. COUNT(DISTINCT ...) sanity check
SELECT
    (SELECT COUNT(*) FROM Customers)                AS total_customers,
    (SELECT COUNT(DISTINCT CustomerID) FROM Orders)  AS distinct_customers_with_orders;

-- 5b. Grouped child-count — does any parent have more than one child row?
--     (expected/normal here: a customer having many orders is NOT an error,
--     it's exactly what 1:many means — this just confirms the shape)
SELECT CustomerID, COUNT(*) AS num_orders
FROM Orders
GROUP BY CustomerID
ORDER BY num_orders DESC;

-- 5c. Orphan check — Order rows whose CustomerID has no matching Customer
SELECT o.OrderID, o.CustomerID
FROM Orders o
WHERE o.CustomerID IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM Customers c WHERE c.CustomerID = o.CustomerID
  );
