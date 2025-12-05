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
from tensorflow.keras.backend import clear_session

# ==========================================
# 1. CONFIGURATION
# ==========================================
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DB_PATH = 'market_data.db'
SYMBOL = 'NIFTY'

# ML Params
LOOKBACK = 60
EPOCHS = 20
BATCH_SIZE = 16
N_FEATURES = 7 # Open, High, Low, Effective_Close, OI, ATR_14, ROC_5

# ==========================================
# 2. DATABASE MANAGEMENT
# ==========================================
def get_db_connection():
    """Context manager for database connection."""
    return sqlite3.connect(DB_PATH)

def migrate_schema(db_path):
    """Ensures tables exist and have required columns."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create Tables
        tables = {
            'nifty_predictions': '''
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT, data_upto_date TEXT, run_time TEXT,
                current_price REAL, predicted_price REAL, expected_move REAL,
                direction TEXT, resistance REAL, support REAL, trading_plan TEXT,
                actual_close REAL, outcome TEXT, failure_reason TEXT, updated_by TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            ''',
            'nifty_daily_forecast': '''
                id INTEGER PRIMARY KEY AUTOINCREMENT, target_date TEXT, 
                pred_open REAL, pred_high REAL, pred_low REAL, pred_close REAL, 
                trend TEXT, run_time DATETIME DEFAULT CURRENT_TIMESTAMP
            ''',
            'nifty_forecast_3month': '''
                id INTEGER PRIMARY KEY AUTOINCREMENT, forecast_date TEXT UNIQUE, 
                pred_open REAL, pred_high REAL, pred_low REAL, pred_close REAL, trend TEXT, 
                base_run_date DATETIME, actual_close REAL, actual_date TEXT,
                Close_Error_Pct REAL, Range_Contained TEXT, Daily_Outcome TEXT, Failure_Reason TEXT
            '''
        }
        
        for table, schema in tables.items():
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")
            
        # Check columns for nifty_predictions
        cursor.execute("PRAGMA table_info(nifty_predictions)")
        existing_cols = {info[1] for info in cursor.fetchall()}
        
        new_cols = {
            'actual_close': 'REAL', 'outcome': 'TEXT',
            'failure_reason': 'TEXT', 'updated_by': 'TEXT'
        }
        
        for col, dtype in new_cols.items():
            if col not in existing_cols:
                print(f"⚙️ Migrating Schema: Adding '{col}' column...")
                cursor.execute(f"ALTER TABLE nifty_predictions ADD COLUMN {col} {dtype}")
        
        conn.commit()

# ==========================================
# 3. OPTIMIZED VALIDATION LOOP
# ==========================================
def validate_past_predictions(db_path):
    """Batch validates past predictions to minimize DB calls."""
    print("\n🕵️ Checking for past predictions to validate...")
    
    with get_db_connection() as conn:
        # Fetch pending predictions
        pending_query = """
            SELECT id, prediction_date, predicted_price 
            FROM nifty_predictions 
            WHERE actual_close IS NULL
        """
        pending = pd.read_sql(pending_query, conn)
        
        if pending.empty:
            print("   > All past predictions are already validated.")
            return

        # Fetch relevant actuals in one go
        min_date = pending['prediction_date'].min()
        actuals_query = f"""
            SELECT Date, Close as actual_close
            FROM index_derivative 
            WHERE Symbol = '{SYMBOL}' AND Instrument = 'FUTIDX' 
            AND Date >= '{min_date}'
        """
        actuals = pd.read_sql(actuals_query, conn)
        
        # Merge data (Vectorized operation)
        merged = pd.merge(pending, actuals, left_on='prediction_date', right_on='Date', how='inner')
        
        if merged.empty:
            print("   > No new actual data available for validation.")
            return

        # Calculate metrics vectorially
        merged['error_pct'] = (abs(merged['actual_close'] - merged['predicted_price']) / merged['actual_close']) * 100
        
        conditions = [
            (merged['error_pct'] < 0.5),
            (merged['error_pct'] < 1.5)
        ]
        choices_outcome = ['PERFECT', 'SUCCESS']
        choices_reason = ['High Accuracy', 'Acceptable Variance']
        
        merged['outcome'] = np.select(conditions, choices_outcome, default='FAILED')
        merged['failure_reason'] = np.select(
            conditions, 
            choices_reason, 
            default=merged['error_pct'].apply(lambda x: f"High Deviation ({x:.2f}%)")
        )
        
        # Bulk Update
        cursor = conn.cursor()
        update_data = merged[['actual_close', 'outcome', 'failure_reason', 'id']].to_records(index=False).tolist()
        # Add 'Validator_Bot' to each record
        update_data = [ (*x, 'Validator_Bot') for x in update_data ]
        
        cursor.executemany('''
            UPDATE nifty_predictions 
            SET actual_close = ?, outcome = ?, failure_reason = ?, updated_by = ?
            WHERE id = ?
        ''', update_data)
        
        conn.commit()
        print(f"   ✅ Batch validated {len(merged)} predictions.")

# ==========================================
# 4. DATA LOADING
# ==========================================
def load_futures_data(db_path, symbol):
    print(f"📂 Loading Historical Futures Data for {symbol}...")
    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found.")
        return None

    with get_db_connection() as conn:
        query = f"""
            SELECT Date, Open, High, Low, Close, Settle_Price, Spot, OI 
            FROM index_derivative 
            WHERE Symbol = '{symbol}' AND Instrument = 'FUTIDX'
            ORDER BY Date ASC
        """
        df = pd.read_sql(query, conn)

    # Fast numeric conversion
    cols = ['Open', 'High', 'Low', 'Close', 'Settle_Price', 'Spot', 'OI']
    for c in cols:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
    
    df['Effective_Close'] = df['Spot'].fillna(df['Close']).fillna(df['Settle_Price'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.dropna(subset=['Effective_Close']).sort_values('Date').drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
    
    # Feature Engineering
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(
        np.abs(df['High'] - df['Effective_Close'].shift(1)), 
        np.abs(df['Low'] - df['Effective_Close'].shift(1))
    ))
    df['ATR_14'] = df['TR'].ewm(span=14, adjust=False).mean()
    df['ROC_5'] = df['Effective_Close'].pct_change(periods=5) * 100
    
    return df.dropna(subset=['ATR_14', 'ROC_5']).reset_index(drop=True)

# ==========================================
# 5. ADAPTIVE LEARNING
# ==========================================
def analyze_recent_performance(db_path):
    with get_db_connection() as conn:
        # Check table existence efficiently
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nifty_forecast_3month'"
        ).fetchone()
        
        if not table_exists:
            return 0.0, "No History"

        query = """
            SELECT actual_close, pred_close 
            FROM nifty_forecast_3month 
            WHERE actual_close IS NOT NULL 
            ORDER BY forecast_date DESC LIMIT 5
        """
        recent = pd.read_sql(query, conn)
        
    if recent.empty:
        return 0.0, "No Validated History"

    bias = (recent['actual_close'] - recent['pred_close']).mean()
    
    if bias > 0: return 0.005, "Underestimation (Bullish Adj)"
    if bias < 0: return -0.005, "Overestimation (Bearish Adj)"
    return 0.0, "Neutral"

# ==========================================
# 6. MODEL & PREDICTION
# ==========================================
def get_smart_money_levels(df):
    data = df.tail(30).reset_index(drop=True)
    curr_price = data['Effective_Close'].iloc[-1]
    
    # Vectorized check for swing points (faster than loop)
    highs = data['High'].values
    lows = data['Low'].values
    
    # Simple local extrema check (window 5)
    # Using scipy argrelextrema would be better, but keeping simple logic:
    resistances = []
    supports = []
    
    for i in range(2, len(data)-2):
        if highs[i] > max(highs[i-1], highs[i-2], highs[i+1], highs[i+2]):
            resistances.append(highs[i])
        if lows[i] < min(lows[i-1], lows[i-2], lows[i+1], lows[i+2]):
            supports.append(lows[i])
            
    valid_res = [r for r in resistances if r > curr_price]
    immed_res = min(valid_res) if valid_res else curr_price + (highs[-1] - lows[-1])
    
    valid_sup = [s for s in supports if s < curr_price]
    immed_sup = max(valid_sup) if valid_sup else curr_price - (highs[-1] - lows[-1])
    
    return immed_sup, immed_res

def predict_next_day_lstm(df):
    raw_values = df[['Open', 'High', 'Low', 'Effective_Close', 'OI', 'ATR_14', 'ROC_5']].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_full = scaler.fit_transform(raw_values)
    
    if len(scaled_full) <= LOOKBACK: return df['Effective_Close'].iloc[-1]

    # Batch creation
    X_train = []
    y_train = []
    
    # Optimize loop: Pre-allocate numpy arrays if possible, but list append is okay for small N
    for i in range(LOOKBACK, len(scaled_full)):
        X_train.append(scaled_full[i-LOOKBACK:i])
        y_train.append(scaled_full[i, 3]) # Target: Effective_Close
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    print("🧠 Training Stacked LSTM Model...")
    clear_session()
    model = Sequential([
        Input(shape=(LOOKBACK, N_FEATURES)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    
    last_sequence = scaled_full[-LOOKBACK:].reshape((1, LOOKBACK, N_FEATURES))
    pred_scaled = model.predict(last_sequence, verbose=0)
    
    dummy = np.zeros((1, N_FEATURES))
    dummy[:, 3] = pred_scaled[0][0]
    return scaler.inverse_transform(dummy)[0][3]

def calculate_ohlc_prediction(current_close, lstm_close, sup_res_data, atr, adj_factor=0.0):
    immed_sup, immed_res = sup_res_data
    lstm_close_adjusted = lstm_close * (1 + adj_factor)
    
    trend = "UP" if lstm_close_adjusted > current_close else "DOWN"
    momentum = (lstm_close_adjusted - current_close) * 0.3
    pred_open = current_close + momentum
    
    if trend == "UP":
        pred_high = min(immed_res, pred_open + (atr * 0.9))
        pred_low = max(immed_sup, pred_open - (atr * 0.4))
    else:
        pred_high = min(immed_res, pred_open + (atr * 0.4))
        pred_low = max(immed_sup, pred_open - (atr * 0.9))
        
    return {
        'Open': pred_open, 
        'High': max(pred_open, lstm_close_adjusted, pred_high), 
        'Low': min(pred_open, lstm_close_adjusted, pred_low), 
        'Close': lstm_close_adjusted, 
        'Trend': trend
    }

def save_daily_prediction(db_path, data, learning_note):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO nifty_daily_forecast (target_date, pred_open, pred_high, pred_low, pred_close, trend) VALUES (?, ?, ?, ?, ?, ?)", 
                       (data['date'], data['o'], data['h'], data['l'], data['c'], data['trend']))
        
        cursor.execute("""
            INSERT INTO nifty_predictions (
                prediction_date, data_upto_date, run_time, current_price, predicted_price, 
                expected_move, direction, resistance, support, trading_plan, updated_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['date'], datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data['last_close'], data['c'], data['c'] - data['last_close'], data['trend'],
            data['h'], data['l'], f"Range: {data['l']:,.0f}-{data['h']:,.0f} | Note: {learning_note}", 'Deep_LSTM_Adaptive'
        ))
        conn.commit()
    print("✅ Saved Forecast to DB.")

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    migrate_schema(DB_PATH)
    validate_past_predictions(DB_PATH)
    df = load_futures_data(DB_PATH, SYMBOL)
    
    if df is not None and len(df) > LOOKBACK:
        last_close = df['Effective_Close'].iloc[-1]
        atr = df['ATR_14'].iloc[-1]
        
        adj_factor, bias_type = analyze_recent_performance(DB_PATH)
        immed_sup, immed_res = get_smart_money_levels(df)
        lstm_pred_close = predict_next_day_lstm(df)
        
        print(f"   > Raw LSTM Target: {lstm_pred_close:.2f}")
        ohlc = calculate_ohlc_prediction(last_close, lstm_pred_close, (immed_sup, immed_res), atr, adj_factor)
        print(f"   > Adjusted Target ({bias_type}): {ohlc['Close']:.2f}")
        
        target_date = df['Date'].iloc[-1] + timedelta(days=1)
        while target_date.weekday() >= 5: target_date += timedelta(days=1)
        
        print("\n" + "="*40)
        print(f"🔮 NIFTY PREDICTION FOR {target_date.date()}")
        print("="*40)
        print(f"   Last Close:     {last_close:,.2f}")
        print(f"   Predicted OPEN: {ohlc['Open']:,.2f}")
        print(f"   Predicted HIGH: {ohlc['High']:,.2f}")
        print(f"   Predicted LOW:  {ohlc['Low']:,.2f}")
        print(f"   Predicted CLOSE:{ohlc['Close']:,.2f}")
        print("-" * 40)
        print(f"   TREND: {ohlc['Trend']} | RANGE: {ohlc['Low']:,.0f} - {ohlc['High']:,.0f}")
        print("="*40)
        
        save_daily_prediction(DB_PATH, {
            'date': target_date.strftime('%Y-%m-%d'), 'o': ohlc['Open'], 'h': ohlc['High'], 
            'l': ohlc['Low'], 'c': ohlc['Close'], 'trend': ohlc['Trend'], 'last_close': last_close 
        }, bias_type)
    else:
        print("Insufficient historical data.")