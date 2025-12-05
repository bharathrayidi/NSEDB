SELECT *
FROM index_derivative
WHERE Symbol = 'NIFTY'
  AND "Date" = (
    SELECT MAX("Date")
    FROM index_derivative
    WHERE Symbol = 'NIFTY'
  );
