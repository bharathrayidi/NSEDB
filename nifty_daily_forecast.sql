SELECT * FROM nifty_daily_forecast
where target_date = (SELECT MAX("target_date")
    FROM nifty_daily_forecast);