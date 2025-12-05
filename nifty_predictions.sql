select * from nifty_predictions
where prediction_date = (SELECT MAX("prediction_date")
    FROM nifty_predictions);