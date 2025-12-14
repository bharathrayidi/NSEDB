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
N_FEATURES = 7  # Open, High, Low, Effective_Close, OI, ATR_14, ROC_5

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
                trend TEXT, actual_open REAL, actual_high REAL, actual_low REAL, actual_close REAL,
                outcome TEXT, error_pct REAL, run_time DATETIME DEFAULT CURRENT_TIMESTAMP
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
        
        # --- MIGRATION: Ensure all columns exist ---
        # nifty_predictions
        cursor.execute("PRAGMA table_info(nifty_predictions)")
        existing_cols = {info[1] for info in cursor.fetchall()}
        new_cols = {
            'actual_close': 'REAL', 'outcome': 'TEXT',
            'failure_reason': 'TEXT', 'updated_by': 'TEXT'
        }
        for col, dtype in new_cols.items():
            if col not in existing_cols:
                print(f"⚙️ Migrating nifty_predictions: Adding '{col}'...")
                cursor.execute(f"ALTER TABLE nifty_predictions ADD COLUMN {col} {dtype}")

        # nifty_daily_forecast
        cursor.execute("PRAGMA table_info(nifty_daily_forecast)")
        daily_cols = {info[1] for info in cursor.fetchall()}
        new_daily_cols = {
            'actual_open': 'REAL', 'actual_high': 'REAL', 'actual_low': 'REAL', 
            'actual_close': 'REAL', 'outcome': 'TEXT', 'error_pct': 'REAL'
        }
        for col, dtype in new_daily_cols.items():
            if col not in daily_cols:
                print(f"⚙️ Migrating nifty_daily_forecast: Adding '{col}'...")
                cursor.execute(f"ALTER TABLE nifty_daily_forecast ADD COLUMN {col} {dtype}")
        
        conn.commit()

# ==========================================
# 3. VALIDATION FUNCTIONS
# ==========================================

def get_market_data_for_validation(conn):
    """Fetch recent market data for validation (Increased limit for backfills)."""
    # Increased limit to 1000 to catch up on large gaps if necessary
    query = f"""
        SELECT Date, Open, High, Low, Close, Spot
        FROM index_derivative 
        WHERE Symbol = '{SYMBOL}' AND Instrument = 'FUTIDX'
        ORDER BY Date DESC LIMIT 1000
    """
    df = pd.read_sql(query, conn)
    if not df.empty:
        df['final_close'] = df['Spot'].fillna(df['Close'])
        df['match_date'] = pd.to_datetime(df['Date']).dt.normalize()
        df = df.drop_duplicates(subset=['match_date'])
    return df

def validate_main_predictions(conn):
    """Validates the main 'nifty_predictions' table."""
    print("\n🕵️ Validating 'nifty_predictions' table...")
    
    pending = pd.read_sql("SELECT id, prediction_date, predicted_price FROM nifty_predictions WHERE actual_close IS NULL", conn)
    if pending.empty:
        print("   > No pending rows.")
        return

    actuals = get_market_data_for_validation(conn)
    if actuals.empty: return

    pending['match_date'] = pd.to_datetime(pending['prediction_date']).dt.normalize()
    merged = pd.merge(pending, actuals, on='match_date', how='inner')

    if merged.empty:
        print("   > No matching dates found yet.")
        return

    merged['error_pct'] = (abs(merged['final_close'] - merged['predicted_price']) / merged['final_close']) * 100
    conditions = [(merged['error_pct'] < 0.5), (merged['error_pct'] < 1.5)]
    merged['outcome'] = np.select(conditions, ['PERFECT', 'SUCCESS'], default='FAILED')
    merged['failure_reason'] = np.select(conditions, ['High Accuracy', 'Acceptable Variance'], default=merged['error_pct'].apply(lambda x: f"High Deviation ({x:.2f}%)"))

    print(f"   📝 Updating {len(merged)} rows in nifty_predictions...")
    update_data = []
    for _, row in merged.iterrows():
        update_data.append((row['final_close'], row['outcome'], row['failure_reason'], 'Backfill_Bot', row['id']))
    
    cursor = conn.cursor()
    cursor.executemany("UPDATE nifty_predictions SET actual_close=?, outcome=?, failure_reason=?, updated_by=? WHERE id=?", update_data)
    conn.commit()

def validate_daily_forecast(conn):
    """Validates the 'nifty_daily_forecast' table (OHLC predictions)."""
    print("\n🕵️ Validating 'nifty_daily_forecast' table...")

    pending = pd.read_sql("SELECT id, target_date, pred_close FROM nifty_daily_forecast WHERE actual_close IS NULL", conn)
    if pending.empty:
        print("   > No pending rows.")
        return

    actuals = get_market_data_for_validation(conn)
    if actuals.empty: return

    pending['match_date'] = pd.to_datetime(pending['target_date']).dt.normalize()
    merged = pd.merge(pending, actuals, on='match_date', how='inner')

    if merged.empty:
        print("   > No matching dates found yet.")
        return

    merged['error_pct'] = (abs(merged['final_close'] - merged['pred_close']) / merged['final_close']) * 100
    merged['outcome'] = np.where(merged['error_pct'] < 1.0, 'HIT', 'MISS')

    print(f"   📝 Updating {len(merged)} rows in nifty_daily_forecast...")
    update_data = []
    for _, row in merged.iterrows():
        update_data.append((
            row['Open'], row['High'], row['Low'], row['final_close'], 
            row['outcome'], row['error_pct'], row['id']
        ))

    cursor = conn.cursor()
    cursor.executemany("""
        UPDATE nifty_daily_forecast 
        SET actual_open=?, actual_high=?, actual_low=?, actual_close=?, outcome=?, error_pct=? 
        WHERE id=?
    """, update_data)
    conn.commit()
    print(f"   ✅ Successfully validated {len(merged)} daily forecasts.")

def run_all_validations(db_path):
    with get_db_connection() as conn:
        validate_main_predictions(conn)
        validate_daily_forecast(conn)

# ==========================================
# 4. DATA LOADING
# ==========================================
def load_futures_data(db_path, symbol):
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
        # Check if table has data first
        try:
            query = """
                SELECT actual_close, pred_close 
                FROM nifty_daily_forecast
                WHERE actual_close IS NOT NULL 
                ORDER BY target_date DESC LIMIT 5
            """
            recent = pd.read_sql(query, conn)
        except:
            return 0.0, "No History"
        
    if recent.empty:
        return 0.0, "No Validated History"

    bias = (recent['actual_close'] - recent['pred_close']).mean()
    
    # Simple logic: If we consistently underpredict, add a small bump
    if bias > 50: return 0.003, "Strong Underest. (Bull+)"
    if bias > 0: return 0.001, "Mild Underest. (Bull+)"
    if bias < -50: return -0.003, "Strong Overest. (Bear-)"
    if bias < 0: return -0.001, "Mild Overest. (Bear-)"
    return 0.0, "Neutral"

# ==========================================
# 6. MODEL & PREDICTION LOGIC
# ==========================================

def train_lstm_model(df_train):
    """Trains the LSTM model on the provided dataframe."""
    raw_values = df_train[['Open', 'High', 'Low', 'Effective_Close', 'OI', 'ATR_14', 'ROC_5']].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_full = scaler.fit_transform(raw_values)
    
    X_train = []
    y_train = []
    
    for i in range(LOOKBACK, len(scaled_full)):
        X_train.append(scaled_full[i-LOOKBACK:i])
        y_train.append(scaled_full[i, 3]) # Target: Effective_Close
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    print(f"   🧠 Training LSTM on {len(X_train)} samples...")
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
    
    return model, scaler

def predict_single_step(model, scaler, df_window):
    """Predicts a single next step using the last LOOKBACK rows of df_window."""
    raw_values = df_window[['Open', 'High', 'Low', 'Effective_Close', 'OI', 'ATR_14', 'ROC_5']].values
    scaled_raw = scaler.transform(raw_values)
    
    last_sequence = scaled_raw[-LOOKBACK:].reshape((1, LOOKBACK, N_FEATURES))
    pred_scaled = model.predict(last_sequence, verbose=0)
    
    dummy = np.zeros((1, N_FEATURES))
    dummy[:, 3] = pred_scaled[0][0]
    return scaler.inverse_transform(dummy)[0][3]

def get_smart_money_levels(df):
    """Calculates Support and Resistance based on pivots."""
    data = df.tail(30).reset_index(drop=True)
    if len(data) < 5: return data['Effective_Close'].iloc[-1], data['Effective_Close'].iloc[-1]

    curr_price = data['Effective_Close'].iloc[-1]
    highs = data['High'].values
    lows = data['Low'].values
    
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
        'Open': pred_open, 'High': max(pred_open, lstm_close_adjusted, pred_high), 
        'Low': min(pred_open, lstm_close_adjusted, pred_low), 'Close': lstm_close_adjusted, 
        'Trend': trend
    }

def save_prediction_to_db(data, learning_note, updated_by):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check if already exists to prevent dupes in backfill
        exists = cursor.execute("SELECT id FROM nifty_daily_forecast WHERE target_date = ?", (data['date'],)).fetchone()
        if exists: return

        cursor.execute("INSERT INTO nifty_daily_forecast (target_date, pred_open, pred_high, pred_low, pred_close, trend) VALUES (?, ?, ?, ?, ?, ?)", 
                       (data['date'], data['o'], data['h'], data['l'], data['c'], data['trend']))
        
        cursor.execute("""
            INSERT INTO nifty_predictions (
                prediction_date, data_upto_date, run_time, current_price, predicted_price, 
                expected_move, direction, resistance, support, trading_plan, updated_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['date'], data['data_upto'], datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data['last_close'], data['c'], data['c'] - data['last_close'], data['trend'],
            data['h'], data['l'], f"Range: {data['l']:,.0f}-{data['h']:,.0f} | Note: {learning_note}", updated_by
        ))
        conn.commit()

# ==========================================
# 7. BACKFILL LOGIC (NEW)
# ==========================================
def get_last_prediction_date():
    with get_db_connection() as conn:
        # Check 'nifty_daily_forecast' as it's the cleaner table
        res = conn.execute("SELECT MAX(target_date) FROM nifty_daily_forecast").fetchone()
        if res and res[0]:
            return pd.to_datetime(res[0])
    return None

def process_missing_predictions(df):
    """Identifies gaps between DB and Market Data, and fills them."""
    print("\n🔄 Checking for missing historical predictions...")
    
    last_pred_date = get_last_prediction_date()
    
    # If DB is empty, start from the first possible prediction day (LOOKBACK days in)
    if last_pred_date is None:
        start_index = LOOKBACK
        print("   > DB is empty. Starting fresh.")
    else:
        # Find the index in DF that corresponds to the last prediction
        # We need to start predicting for the day *after* last_pred_date
        # So we look for dates > last_pred_date
        future_dates = df[df['Date'] > last_pred_date]
        if future_dates.empty:
            print("   > DB is up to date with available market data.")
            return
        
        # The first missing index is the first row of future_dates
        start_index = future_dates.index[0]
        print(f"   > Found gap starting from {future_dates['Date'].iloc[0].date()}")

    # Training Strategy: Train ONCE on data available BEFORE the gap starts to avoid data leakage.
    # We define the training set as everything up to start_index - 1
    train_cutoff = start_index
    if train_cutoff < LOOKBACK: 
        print("   > Not enough data to train.")
        return

    print("   > Training model for backfill/catch-up...")
    df_train = df.iloc[:train_cutoff]
    model, scaler = train_lstm_model(df_train)

    # Iterate through missing days
    # For each day 'i', we predict using window [i-LOOKBACK : i]
    # This simulates a "rolling" prediction without retraining the weights every single day (for speed)
    count = 0
    for i in range(start_index, len(df)):
        target_date = df['Date'].iloc[i]
        
        # Data available UP TO the previous day
        window = df.iloc[i-LOOKBACK : i]
        last_close = window['Effective_Close'].iloc[-1]
        atr = window['ATR_14'].iloc[-1]
        
        # Predict
        lstm_pred = predict_single_step(model, scaler, window)
        immed_sup, immed_res = get_smart_money_levels(window)
        
        ohlc = calculate_ohlc_prediction(last_close, lstm_pred, (immed_sup, immed_res), atr, adj_factor=0.0)
        
        save_prediction_to_db({
            'date': target_date.strftime('%Y-%m-%d'),
            'data_upto': window['Date'].iloc[-1].strftime('%Y-%m-%d'),
            'o': ohlc['Open'], 'h': ohlc['High'], 'l': ohlc['Low'], 'c': ohlc['Close'],
            'trend': ohlc['Trend'], 'last_close': last_close
        }, "Backfilled", "Backfill_Bot")
        
        print(f"   > Backfilled: {target_date.date()} | Pred: {ohlc['Close']:.2f} | Actual: {df['Effective_Close'].iloc[i]:.2f}")
        count += 1
        
    print(f"✅ Backfill complete. Added {count} entries.")

# ==========================================
# 8. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Market Prediction Engine...")
    
    # 1. Setup
    migrate_schema(DB_PATH)
    
    # 2. Load Data
    df = load_futures_data(DB_PATH, SYMBOL)
    
    if df is not None and len(df) > LOOKBACK:
        
        # 3. BACKFILL GAPS (The new feature)
        # This will predict and save any past dates present in 'df' but missing in DB
        process_missing_predictions(df)
        
        # 4. VALIDATE EVERYTHING
        # Now that gaps are filled, run validation to mark them HIT/MISS
        run_all_validations(DB_PATH)
        
        # 5. PREDICT FUTURE (NEXT TRADING DAY)
        # Now we calculate the bias based on the FULL history (including just validated backfills)
        print("\n🔮 Generating Future Forecast...")
        adj_factor, bias_type = analyze_recent_performance(DB_PATH)
        
        # We retrain the model on the COMPLETE dataset now for the best future prediction
        model, scaler = train_lstm_model(df)
        
        last_window = df.tail(LOOKBACK)
        last_close = df['Effective_Close'].iloc[-1]
        atr = df['ATR_14'].iloc[-1]
        immed_sup, immed_res = get_smart_money_levels(df)
        
        lstm_pred_close = predict_single_step(model, scaler, last_window)
        
        print(f"   > Raw LSTM Target: {lstm_pred_close:.2f}")
        ohlc = calculate_ohlc_prediction(last_close, lstm_pred_close, (immed_sup, immed_res), atr, adj_factor)
        print(f"   > Adjusted Target ({bias_type}): {ohlc['Close']:.2f}")
        
        target_date = df['Date'].iloc[-1] + timedelta(days=1)
        while target_date.weekday() >= 5: target_date += timedelta(days=1)
        
        print("\n" + "="*40)
        print(f"📢 NIFTY PREDICTION FOR {target_date.date()}")
        print("="*40)
        print(f"   Last Close:     {last_close:,.2f}")
        print(f"   Predicted OPEN: {ohlc['Open']:,.2f}")
        print(f"   Predicted HIGH: {ohlc['High']:,.2f}")
        print(f"   Predicted LOW:  {ohlc['Low']:,.2f}")
        print(f"   Predicted CLOSE:{ohlc['Close']:,.2f}")
        print("-" * 40)
        print(f"   TREND: {ohlc['Trend']} | RANGE: {ohlc['Low']:,.0f} - {ohlc['High']:,.0f}")
        print("="*40)
        
        save_prediction_to_db({
            'date': target_date.strftime('%Y-%m-%d'),
            'data_upto': df['Date'].iloc[-1].strftime('%Y-%m-%d'),
            'o': ohlc['Open'], 'h': ohlc['High'], 'l': ohlc['Low'], 'c': ohlc['Close'],
            'trend': ohlc['Trend'], 'last_close': last_close 
        }, bias_type, "Deep_LSTM_Live")
        
        print("✅ Forecast Saved.")
    else:
        print("❌ Insufficient historical data.")