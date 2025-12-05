import sqlite3
import pandas as pd
import numpy as np
import os
import warnings
import time
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_absolute_percentage_error as mape
from tensorflow.keras.backend import clear_session 

# ==========================================
# 1. CONFIGURATION
# ==========================================
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DB_PATH = 'market_data.db'
SYMBOL = 'NIFTY'

# ML Params
LOOKBACK = 30
EPOCHS = 10 
BATCH_SIZE = 16
N_FEATURES = 7 

# Time Split Parameters
TRAIN_DAYS = 84 
VALIDATION_DAYS = 63 

# GLOBAL MODEL DEFINITION
LSTM_MODEL = None 
INITIAL_WEIGHTS = None

# ==========================================
# 2. DATABASE MANAGEMENT
# ==========================================
def migrate_schema(db_path):
    """Ensures all required tables have the correct schema."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nifty_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, prediction_date TEXT, data_upto_date TEXT, 
                run_time TEXT, current_price REAL, predicted_price REAL, expected_move REAL, 
                direction TEXT, resistance REAL, support REAL, trading_plan TEXT, actual_close REAL, 
                outcome TEXT, failure_reason TEXT, updated_by TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nifty_forecast_3month (
                id INTEGER PRIMARY KEY AUTOINCREMENT, forecast_date TEXT UNIQUE, 
                pred_open REAL, pred_high REAL, pred_low REAL, pred_close REAL, trend TEXT, 
                base_run_date DATETIME, 
                actual_close REAL, actual_date TEXT,
                Close_Error_Pct REAL, Range_Contained TEXT, Daily_Outcome TEXT, Failure_Reason TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nifty_model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                train_start_date TEXT, train_end_date TEXT, 
                validation_start_date TEXT, validation_end_date TEXT,
                train_window_days INTEGER, validation_window_days INTEGER, 
                mape REAL, status TEXT, failure_reason TEXT, improvement_plan TEXT
            )
        ''')

        tables_to_check = {
            'nifty_forecast_3month': {
                'Close_Error_Pct': 'REAL', 'Range_Contained': 'TEXT', 
                'Daily_Outcome': 'TEXT', 'Failure_Reason': 'TEXT'
            },
            'nifty_model_performance': {
                'train_start_date': 'TEXT', 'train_end_date': 'TEXT', 
                'validation_start_date': 'TEXT', 'validation_end_date': 'TEXT', 
                'validation_window_days': 'INTEGER'
            }
        }
        
        for table, new_cols in tables_to_check.items():
            cursor.execute(f"PRAGMA table_info({table})")
            existing_cols = [info[1] for info in cursor.fetchall()]
            for col, dtype in new_cols.items():
                if col not in existing_cols:
                    try:
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")
                    except sqlite3.OperationalError:
                        pass 

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Schema Migration Error: {e}")

# ==========================================
# 4. DATA LOADING & PREPARATION
# ==========================================
def load_futures_data(db_path, symbol):
    """Loads and preprocesses futures data with fallbacks for close price and adds features."""
    print(f"📂 Loading Historical Futures Data for {symbol}...")
    try:
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT Date, Open, High, Low, Close, Settle_Price, Spot, OI 
            FROM index_derivative 
            WHERE Symbol = '{symbol}' AND Instrument = 'FUTIDX'
            ORDER BY Date ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Settle_Price', 'Spot', 'OI']
        for c in numeric_cols:
            if c in df.columns and df[c].dtype == 'object':
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
        
        df['Effective_Close'] = df['Spot'].fillna(df['Close']).fillna(df['Settle_Price'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna(subset=['Effective_Close']).sort_values('Date')
        df = df.drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
        
        # --- FEATURE ENGINEERING ---
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(
            np.abs(df['High'] - df['Effective_Close'].shift(1)), 
            np.abs(df['Low'] - df['Effective_Close'].shift(1))
        ))
        df['ATR_14'] = df['TR'].ewm(span=14, adjust=False).mean()
        df['ROC_5'] = df['Effective_Close'].pct_change(periods=5) * 100
        
        df = df.dropna(subset=['ATR_14', 'ROC_5']).reset_index(drop=True)

        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def prepare_data_for_extended_prediction(df):
    """Splits data into Train (84 days), Validation (63 days) for backtesting."""
    
    if len(df) < TRAIN_DAYS + VALIDATION_DAYS:
        print(f"❌ Insufficient data. Need at least {TRAIN_DAYS + VALIDATION_DAYS} days.")
        return None, None, None
    
    validation_end_idx = len(df)
    validation_start_idx = validation_end_idx - VALIDATION_DAYS
    train_start_idx = validation_start_idx - TRAIN_DAYS
    
    train_df = df.iloc[train_start_idx:validation_start_idx].copy().reset_index(drop=True)
    validation_df = df.iloc[validation_start_idx:validation_end_idx].copy().reset_index(drop=True)
    
    validation_dates = validation_df['Date'].tolist()

    print(f"Split complete (Backtesting Mode):")
    print(f"  > Train Period: {train_df['Date'].iloc[0].date()} to {train_df['Date'].iloc[-1].date()} ({len(train_df)} days)")
    print(f"  > Validation Period (Past Forecast): {validation_df['Date'].iloc[0].date()} to {validation_df['Date'].iloc[-1].date()} ({len(validation_df)} days)")

    return train_df, validation_df, validation_dates

# ==========================================
# 5. SMART MONEY LEVELS
# ==========================================
def get_smart_money_levels(df):
    """Finds levels for the LIVE prediction using only Price Action."""
    data = df.tail(30).reset_index(drop=True)
    curr_price = data['Effective_Close'].iloc[-1] 
    resistances, supports = [], []
    
    for i in range(2, len(data)-2):
        if data['High'][i] > max(data['High'][i-1], data['High'][i-2], data['High'][i+1], data['High'][i+2]):
            resistances.append(data['High'][i])
        if data['Low'][i] < min(data['Low'][i-1], data['Low'][i-2], data['Low'][i+1], data['Low'][i+2]):
            supports.append(data['Low'][i])
            
    valid_res = [r for r in resistances if r > curr_price]
    immed_res = min(valid_res) if valid_res else curr_price + (data['High'].iloc[-1] - data['Low'].iloc[-1])
    
    valid_sup = [s for s in supports if s < curr_price]
    immed_sup = max(valid_sup) if valid_sup else curr_price - (data['High'].iloc[-1] - data['Low'].iloc[-1])
    
    return immed_sup, immed_res

# ==========================================
# 6. LSTM PREDICTION
# ==========================================
def predict_next_day_lstm(df_for_model, model):
    """Trains and predicts tomorrow's Effective_Close using LSTM."""
    
    if INITIAL_WEIGHTS is not None:
        model.set_weights(INITIAL_WEIGHTS)
    
    # Check for NaNs before training to avoid errors
    if df_for_model[['ATR_14', 'ROC_5']].isnull().values.any():
        df_for_model = df_for_model.fillna(method='ffill').fillna(method='bfill')

    raw_values = df_for_model[['Open', 'High', 'Low', 'Effective_Close', 'OI', 'ATR_14', 'ROC_5']].values
    target_index = 3 
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_full = scaler.fit_transform(raw_values)
    
    X_train, y_train = [], []
    if len(scaled_full) <= LOOKBACK: return df_for_model['Effective_Close'].iloc[-1]

    for i in range(LOOKBACK, len(scaled_full)):
        X_train.append(scaled_full[i-LOOKBACK:i])
        y_train.append(scaled_full[i, target_index])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    
    last_sequence = scaled_full[-LOOKBACK:].reshape((1, LOOKBACK, N_FEATURES)) 
    pred_scaled = model.predict(last_sequence, verbose=0) 
    
    dummy = np.zeros((1, N_FEATURES))
    dummy[:, target_index] = pred_scaled[0][0]
    return scaler.inverse_transform(dummy)[0][target_index] 

# ==========================================
# 7. FINAL PREDICTION LOGIC & ITERATIVE FORECAST
# ==========================================
def calculate_ohlc_prediction(current_close, lstm_close, sup_res_data, atr):
    """Combines ML Trend with calculated S/R levels to predict O/H/L/C."""
    immed_sup, immed_res = sup_res_data
    trend = "UP" if lstm_close > current_close else "DOWN"
    momentum = (lstm_close - current_close) * 0.3
    pred_open = current_close + momentum
    
    if trend == "UP":
        pred_high = min(immed_res, pred_open + (atr * 0.9))
        pred_low = max(immed_sup, pred_open - (atr * 0.4))
    else:
        pred_high = min(immed_res, pred_open + (atr * 0.4))
        pred_low = max(immed_sup, pred_open - (atr * 0.9))
        
    pred_close = lstm_close
    
    final_high = max(pred_open, pred_close, pred_high)
    final_low = min(pred_open, pred_close, pred_low)
    
    return {
        'Open': pred_open, 'High': final_high, 'Low': final_low, 
        'Close': pred_close, 'Trend': trend
    }

def predict_period(base_df, period_dates, period_data):
    """Iteratively predicts the next N days and compares against actual data."""
    
    df_history = base_df.copy()
    results = []
    
    print(f"\n🔮 Starting Validation Period Prediction (Backtesting) for {len(period_dates)} days...")
    
    for i, target_date in enumerate(period_dates):
        
        lstm_pred_close = predict_next_day_lstm(df_history, LSTM_MODEL)
        
        # Safety check: if model output is NaN, carry forward last close
        if np.isnan(lstm_pred_close):
             lstm_pred_close = df_history['Effective_Close'].iloc[-1]

        last_close = df_history['Effective_Close'].iloc[-1]
        
        atr = df_history['ATR_14'].iloc[-1] 
        immed_sup, immed_res = get_smart_money_levels(df_history)
        
        ohlc = calculate_ohlc_prediction(last_close, lstm_pred_close, (immed_sup, immed_res), atr)
        
        actual_row = period_data.iloc[i].copy()
        actual_close = actual_row['Effective_Close']

        result = {
            'forecast_date': target_date.strftime('%Y-%m-%d'),
            'pred_open': ohlc['Open'],
            'pred_high': ohlc['High'],
            'pred_low': ohlc['Low'],
            'pred_close': ohlc['Close'],
            'trend': ohlc['Trend'],
            'actual_close': actual_close, 
            'actual_date': target_date.strftime('%Y-%m-%d')
        }
        
        # Prepare for next iteration
        new_data_row = actual_row.to_dict()
        new_data_row['TR'] = np.maximum(new_data_row['High'] - new_data_row['Low'], 
                                        np.maximum(np.abs(new_data_row['High'] - last_close), 
                                                   np.abs(new_data_row['Low'] - last_close)))
        
        df_history = pd.concat([df_history, pd.DataFrame([new_data_row])], ignore_index=True)

        # Recalculate rolling features
        df_history['ATR_14'] = df_history['TR'].ewm(span=14, adjust=False).mean()
        df_history['ROC_5'] = df_history['Effective_Close'].pct_change(periods=5) * 100
        
        # Keep buffer
        df_history = df_history.iloc[- (TRAIN_DAYS + LOOKBACK + 14):].copy().reset_index(drop=True)

        results.append(result)
        
    return results

# ==========================================
# 8. UTILS & SAVE
# ==========================================
def get_next_trading_day(date_obj):
    """Finds the next trading day (skips Sat/Sun)."""
    date_obj += timedelta(days=1)
    while date_obj.weekday() >= 5: # 5=Sat, 6=Sun
        date_obj += timedelta(days=1)
    return date_obj

def save_extended_forecast(db_path, validation_results, run_date):
    """Saves the 3-month forecast/validation results to the dedicated table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    ACCURACY_THRESHOLD_PCT = 1.5 
    
    for result in validation_results:
        # Check for NaN in values before saving
        if pd.isna(result['pred_close']) or pd.isna(result['actual_close']):
            continue

        close_error_pct = (abs(result['actual_close'] - result['pred_close']) / result['actual_close']) * 100
        range_contained = 'YES' if (result['actual_close'] >= result['pred_low']) and (result['actual_close'] <= result['pred_high']) else 'NO'
        daily_outcome = 'SUCCESS' if (close_error_pct <= ACCURACY_THRESHOLD_PCT) or (range_contained == 'YES') else 'FAILED'
        
        failure_reason = None
        if daily_outcome == 'FAILED':
            close_reason = f"Close Error ({close_error_pct:.2f}%) exceeded {ACCURACY_THRESHOLD_PCT}%." if close_error_pct > ACCURACY_THRESHOLD_PCT else "Close prediction failed accuracy threshold."
            range_reason = f"Actual Close ({result['actual_close']:.2f}) missed predicted range [{result['pred_low']:.2f}-{result['pred_low']:.2f}]." if range_contained == 'NO' else "Range check failed."
            failure_reason = f"Close Prediction: {close_reason} | Range: {range_reason}"

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO nifty_forecast_3month (
                    forecast_date, pred_open, pred_high, pred_low, pred_close, trend, base_run_date, actual_close, actual_date, 
                    Close_Error_Pct, Range_Contained, Daily_Outcome, Failure_Reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['forecast_date'], result['pred_open'], result['pred_high'], result['pred_low'], 
                result['pred_close'], result['trend'], run_date, result['actual_close'], result['actual_date'],
                close_error_pct, range_contained, daily_outcome, failure_reason
            ))
        except Exception as e:
            print(f"Error saving forecast for {result['forecast_date']}: {e}")

    conn.commit()
    conn.close()
    print(f"\n✅ Saved {len(validation_results)} validation forecasts to nifty_forecast_3month with analysis.")

def analyze_failure_and_improve(mape_score):
    """Generates a failure analysis and improvement plan based on MAPE."""
    if mape_score is None:
        status = "DATA_ERROR"
        reason = "Validation period could not be run due to insufficient data or null values."
        plan = "Verify database integrity and data volume."
    elif mape_score < 0.8:
        status = "PERFECT"
        reason = f"Excellent predictive power (MAPE: {mape_score:.2f}%). Model is highly reliable."
        plan = "Explore using this model for multi-day predictions (e.g., 2 or 3 steps ahead) instead of just 1-day ahead."
    elif mape_score < 1.5:
        status = "SUCCESS"
        reason = f"Solid model performance (MAPE: {mape_score:.2f}%). Predicts next day price with reasonable accuracy."
        plan = "Explore more advanced Stacked/Bidirectional LSTM architectures or higher-frequency data."
    else:
        status = "FAILED"
        reason = f"High prediction error (MAPE: {mape_score:.2f}%). The model failed to generalize on the 3-month validation set."
        plan = f"INCREASE LOOKBACK window (try 90 days), INCREASE EPOCHS (try 20), or incorporate external macro features (e.g., VIX, USDINR, global index data) to capture market momentum shifts."

    return status, reason, plan

def save_model_performance(db_path, train_start, train_end, validation_start, validation_end, train_days, validation_days, mape_score, status, reason, plan):
    """Saves the performance metrics and analysis to the dedicated table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO nifty_model_performance (
            train_start_date, train_end_date, validation_start_date, validation_end_date,
            train_window_days, validation_window_days, mape, status, failure_reason, improvement_plan
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        train_start, train_end, validation_start, validation_end,
        train_days, validation_days, mape_score, status, reason, plan
    ))
    
    conn.commit()
    conn.close()
    print("\n✅ Saved Model Performance Summary.")

def create_model_architecture():
    """Defines and compiles the STACKED LSTM model structure."""
    clear_session() 
    model = Sequential([
        Input(shape=(LOOKBACK, N_FEATURES)), 
        LSTM(50, return_sequences=True), 
        Dropout(0.2),
        LSTM(50), 
        Dropout(0.2),
        Dense(25), 
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ==========================================
# 9. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # --- 0. Initialize Global Model ---
    LSTM_MODEL = create_model_architecture()
    INITIAL_WEIGHTS = LSTM_MODEL.get_weights() 
    
    start_time = time.time()
    RUN_DATE = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 1. Setup & Validation
    migrate_schema(DB_PATH) 
    df = load_futures_data(DB_PATH, SYMBOL)
    
    MIN_REQUIRED_DAYS = TRAIN_DAYS + VALIDATION_DAYS + LOOKBACK 
    if df is None or len(df) < MIN_REQUIRED_DAYS:
        print(f"❌ Error: Insufficient data. Need at least {MIN_REQUIRED_DAYS} data points after feature calculation.")
        print(f"Current valid data points available: {len(df) if df is not None else 0}")
        print("Suggestion: Increase historical data in your database or reduce TRAIN_DAYS/LOOKBACK in CONFIGURATION.")
        exit()

    train_df, validation_df, validation_dates = prepare_data_for_extended_prediction(df)

    if train_df is None:
        exit()
        
    # --- 2. VALIDATION PERIOD (Past Forecast: 3 MONTHS) ---
    validation_results = predict_period(train_df, validation_dates, period_data=validation_df)
    
    # Calculate MAPE safely
    actual_closes = [r['actual_close'] for r in validation_results]
    predicted_closes = [r['pred_close'] for r in validation_results]
    
    # Filter out NaNs
    clean_actuals = []
    clean_preds = []
    for a, p in zip(actual_closes, predicted_closes):
        if not pd.isna(a) and not pd.isna(p):
            clean_actuals.append(a)
            clean_preds.append(p)
            
    if len(clean_actuals) < len(actual_closes):
        print(f"⚠️ Warning: Dropped {len(actual_closes) - len(clean_actuals)} NaN predictions/actuals before MAPE calculation.")

    mape_score = None
    if len(clean_actuals) > 0:
        mape_score = mape(clean_actuals, clean_preds) * 100
        
    # --- 3. ANALYZE AND SAVE PERFORMANCE ---
    train_start = train_df['Date'].iloc[0].strftime('%Y-%m-%d')
    train_end = train_df['Date'].iloc[-1].strftime('%Y-%m-%d')
    validation_start = validation_df['Date'].iloc[0].strftime('%Y-%m-%d')
    validation_end = validation_df['Date'].iloc[-1].strftime('%Y-%m-%d')
    
    status, reason, plan = analyze_failure_and_improve(mape_score)
    
    save_model_performance(
        DB_PATH, train_start, train_end, validation_start, validation_end,
        TRAIN_DAYS, VALIDATION_DAYS, mape_score, status, reason, plan
    )
    
    # --- 4. SAVE VALIDATION RESULTS ---
    save_extended_forecast(DB_PATH, validation_results, RUN_DATE)
    
    # --- 5. FINAL OUTPUT ---
    print("\n" + "="*70)
    print("📈 LSTM NIFTY 3-MONTH BACKTEST VALIDATION SUMMARY (Deep Learning)")
    print("="*70)
    print(f"📊 Model Trained on {TRAIN_DAYS} days: {train_start} to {train_end}")
    print(f"🧪 Model Validated on {VALIDATION_DAYS} days: {validation_start} to {validation_end}")
    print("-" * 70)
    print(f"🎯 VALIDATION PERFORMANCE (MAPE): {mape_score:.2f}%") 
    print(f"STATUS: **{status}**")
    print("-" * 70)
    print(f"💡 WHY IT FAILED/SUCCEEDED: {reason}")
    print(f"🛠️ HOW TO IMPROVE: {plan}")
    print("="*70)
    print(f"Run time: {time.time() - start_time:.2f} seconds")