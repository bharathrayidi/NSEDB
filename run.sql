delete FROM nifty_daily_forecast
where target_date = (
        SELECT MAX("target_date")
        FROM nifty_daily_forecast
    );
delete FROM nifty_predictions
where prediction_date = (
        SELECT MAX("prediction_date")
        FROM nifty_predictions
    );
-- DROP TABLE stock_derivative_predictions;
-- DROP TABLE stock_cash_prediction;
-- delete FROM nifty_forecast_3month