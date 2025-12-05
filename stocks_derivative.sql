select * from stocks_derivative
where Date = (SELECT MAX("date")
    FROM stocks_derivative);