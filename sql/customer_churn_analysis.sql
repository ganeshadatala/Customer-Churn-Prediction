USE CustomerChurnDB;
GO

-- Query 1: View first 10 customers
SELECT TOP 10 *
FROM dbo.TelcoChurn;
GO

-- Query 2: Total customers
SELECT COUNT(*) AS TotalCustomers
FROM dbo.TelcoChurn;
GO

-- Query 3: Churn count
SELECT
Churn,
COUNT(*) AS TotalCustomers
FROM dbo.TelcoChurn
GROUP BY Churn;
GO

-- Query 4: Calculate churn percentage
SELECT
Churn,
COUNT(*) AS CustomerCount,
CAST(
100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
AS DECIMAL(5,2)
) AS Percentage
FROM dbo.TelcoChurn
GROUP BY Churn;
GO

-- Query 5: Churn by Contract
SELECT
Contract,
COUNT(*) AS TotalCustomers,
SUM(
CASE
WHEN Churn = 1 THEN 1
ELSE 0
END
) AS ChurnedCustomers
FROM dbo.TelcoChurn
GROUP BY Contract;
GO

-- Query 6: Payment Method Churn Analysis
SELECT
PaymentMethod,
COUNT(*) AS TotalCustomers,
SUM(
CASE
WHEN Churn = 1 THEN 1
ELSE 0
END
) AS ChurnedCustomers
FROM dbo.TelcoChurn
GROUP BY PaymentMethod;
GO
