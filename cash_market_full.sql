-- select * from cash_market_full
-- where Date = (SELECT MAX("date")
--     FROM cash_market_full);

select * from cash_market_full
where Symbol = 'APOLLOHOSP'
ORDER BY Date DESC

