import sqlite3
import pandas as pd
import numpy as np
import os
import warnings
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

# ML Params (Optimized for speed across multiple stocks)
LOOKBACK = 30       # Shorter lookback for individual stocks
EPOCHS = 10         # Fewer epochs for speed
BATCH_SIZE = 16
N_FEATURES = 6      # Open, High, Low, Close, Volume, ROC

# ==========================================
# 2. DATABASE & SCHEMA
# ==========================================
def get_db_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Create final predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_stock_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT,
                symbol TEXT,
                current_price REAL,
                predicted_price REAL,
                signal_source TEXT,
                ml_direction TEXT,
                final_call TEXT,
                nearest_ob REAL,
                confidence TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

# ==========================================
# 3. ORDER BLOCK LOGIC (The "Wick" Check)
# ==========================================
def find_nearest_order_block(df, current_price):
    """
    Scans the last 20 days to find the nearest valid 'Wick Zone' (Order Block).
    Returns (Price Level, Type)
    """
    # Calculate Wicks
    df['Body'] = abs(df['Close'] - df['Open'])
    df['LowerWick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['UpperWick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    
    # Identify Zones (Factor 1.5x body)
    bullish_obs = df[df['LowerWick'] > (df['Body'] * 1.5)]
    bearish_obs = df[df['UpperWick'] > (df['Body'] * 1.5)]
    
    nearest_support = 0.0
    nearest_resistance = 0.0
    
    # Find Support (Bullish OB below price)
    valid_supports = bullish_obs[bullish_obs['Low'] < current_price]
    if not valid_supports.empty:
        # The 'Low' of the wick candle is the strong support
        nearest_support = valid_supports['Low'].iloc[-1] 
        
    # Find Resistance (Bearish OB above price)
    valid_res = bearish_obs[bearish_obs['High'] > current_price]
    if not valid_res.empty:
        # The 'High' of the wick candle is the strong resistance
        nearest_resistance = valid_res['High'].iloc[-1]
        
    return nearest_support, nearest_resistance

# ==========================================
# 4. LSTM MODEL (Per Stock)
# ==========================================
def predict_stock_lstm(df):
    """Trains a quick LSTM model for a specific stock."""
    # Feature Engineering
    df['ROC'] = df['Close'].pct_change(5)
    data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'ROC']].dropna()
    
    if len(data) < LOOKBACK + 10: return None
    
    raw_values = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_full = scaler.fit_transform(raw_values)
    
    X_train, y_train = [], []
    for i in range(LOOKBACK, len(scaled_full)):
        X_train.append(scaled_full[i-LOOKBACK:i])
        y_train.append(scaled_full[i, 3]) # Target: Close
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Clear memory from previous stock loop
    clear_session()
    
    # Lightweight Model
    model = Sequential([
        Input(shape=(LOOKBACK, N_FEATURES)),
        LSTM(30, return_sequences=False), # Single layer for speed
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    
    # Predict Next Day
    last_sequence = scaled_full[-LOOKBACK:].reshape((1, LOOKBACK, N_FEATURES))
    pred_scaled = model.predict(last_sequence, verbose=0)
    
    # Inverse Transform
    dummy = np.zeros((1, N_FEATURES))
    dummy[:, 3] = pred_scaled[0][0]
    final_pred = scaler.inverse_transform(dummy)[0][3]
    
    return final_pred

# ==========================================
# 5. MAIN LOGIC
# ==========================================
def run_live_predictions():
    init_db()
    conn = get_db_connection()
    
    print("🚀 STARTING LIVE STOCK PREDICTION ENGINE")
    print("-" * 50)
    
    # 1. Get List of Stocks to Trade (Filtered from previous step)
    try:
        candidates = pd.read_sql("SELECT * FROM stocks_to_trade WHERE Trade_Signal != 'Neutral'", conn)
    except:
        print("❌ 'stocks_to_trade' table not found. Run 'generate_signals.py' first.")
        return

    if candidates.empty:
        print("⚠️ No trade signals found in database. Exiting.")
        return

    print(f"📋 Analyzing {len(candidates)} candidate stocks...")
    
    results = []
    
    # 2. Iterate through each stock
    for i, row in candidates.iterrows():
        symbol = row['Symbol']
        signal = row['Trade_Signal'] # "BUY (Wick Rejection)"
        
        print(f"\n[{i+1}/{len(candidates)}] Processing {symbol} ({signal})...")
        
        # Load Data
        df = pd.read_sql(f"SELECT * FROM cash_market_full WHERE Symbol='{symbol}' ORDER BY Date", conn)
        df['Date'] = pd.to_datetime(df['Date'])
        
        if df.empty:
            print(f"   ⚠️ No data found for {symbol}")
            continue
            
        current_close = df['Close'].iloc[-1]
        
        # A. Find Nearest Levels (Order Blocks)
        sup, res = find_nearest_order_block(df, current_close)
        
        # B. Run LSTM Prediction
        print("   🧠 Training AI Model...")
        pred_close = predict_stock_lstm(df)
        
        if pred_close is None:
            print("   ⚠️ Not enough data for ML.")
            continue
            
        # C. Logic Confirmation
        ml_move = "UP" if pred_close > current_close else "DOWN"
        ml_pct = ((pred_close - current_close) / current_close) * 100
        
        final_call = "AVOID"
        confidence = "Low"
        key_level = 0.0
        
        # Logic: Signal + ML Agreement + Order Block proximity
        if "BUY" in signal:
            key_level = sup
            if ml_move == "UP":
                final_call = "CONFIRMED BUY"
                confidence = "High" if abs(current_close - sup) / current_close < 0.02 else "Medium"
            else:
                final_call = "WEAK BUY (ML Bearish)"
                
        elif "SELL" in signal:
            key_level = res
            if ml_move == "DOWN":
                final_call = "CONFIRMED SELL"
                confidence = "High" if abs(current_close - res) / current_close < 0.02 else "Medium"
            else:
                final_call = "WEAK SELL (ML Bullish)"

        print(f"   🎯 Prediction: {current_close:.2f} -> {pred_close:.2f} ({ml_move})")
        print(f"   ✅ Verdict: {final_call} (Confidence: {confidence})")
        
        results.append((
            datetime.now().strftime('%Y-%m-%d'),
            symbol, current_close, pred_close, signal, ml_move, 
            final_call, key_level, confidence
        ))

    # 3. Save Results
    if results:
        print("\n💾 Saving predictions to 'live_stock_predictions'...")
        cursor = conn.cursor()
        # Clear old today data to avoid dupes
        cursor.execute("DELETE FROM live_stock_predictions WHERE prediction_date = ?", (datetime.now().strftime('%Y-%m-%d'),))
        
        cursor.executemany("""
            INSERT INTO live_stock_predictions 
            (prediction_date, symbol, current_price, predicted_price, signal_source, ml_direction, final_call, nearest_ob, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, results)
        conn.commit()
        print("✅ Done!")
    else:
        print("⚠️ No valid predictions generated.")
        
    conn.close()

if __name__ == "__main__":
    run_live_predictions()