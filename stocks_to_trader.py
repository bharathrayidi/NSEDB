import sqlite3
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
DB_PATH = 'market_data.db'
LOOKBACK_DAYS = 20  # How far back to look for support/resistance blocks

# ==========================================
# DATABASE CONNECTION
# ==========================================
def get_db_connection():
    return sqlite3.connect(DB_PATH)

def init_signal_table():
    """Creates the table to store potential trade setups."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS stocks_to_trade')
        cursor.execute('''
            CREATE TABLE stocks_to_trade (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Symbol TEXT,
                Date TEXT,
                Close REAL,
                Trade_Signal TEXT,
                Zone_Price REAL,
                Strength TEXT
            )
        ''')
        conn.commit()

# ==========================================
# SIGNAL LOGIC (WICK REJECTIONS)
# ==========================================
def analyze_stock(df):
    """
    Analyzes a single stock for:
    1. Wick Rejections (Hammer/Shooting Star)
    2. Proximity to recent Order Blocks (Support/Resistance)
    """
    if len(df) < LOOKBACK_DAYS: return None

    last_row = df.iloc[-1]
    curr_price = last_row['Close']
    
    # --- 1. Identify Wick Rejections ---
    body_size = abs(last_row['Close'] - last_row['Open'])
    lower_wick = min(last_row['Open'], last_row['Close']) - last_row['Low']
    upper_wick = last_row['High'] - max(last_row['Open'], last_row['Close'])
    
    signal = "Neutral"
    zone_price = 0.0
    strength = "None"

    # Bullish Hammer (Long Lower Wick) at local lows
    if lower_wick > (body_size * 2) and lower_wick > upper_wick:
        # Check if we are near a recent low (Support Check)
        recent_low = df['Low'].iloc[-10:].min()
        if abs(last_row['Low'] - recent_low) / recent_low < 0.01: # Within 1% of recent low
            signal = "BUY (Hammer Rejection)"
            zone_price = last_row['Low']
            strength = "High"

    # Bearish Shooting Star (Long Upper Wick) at local highs
    elif upper_wick > (body_size * 2) and upper_wick > lower_wick:
        # Check if we are near a recent high (Resistance Check)
        recent_high = df['High'].iloc[-10:].max()
        if abs(last_row['High'] - recent_high) / recent_high < 0.01: # Within 1% of recent high
            signal = "SELL (Shooting Star)"
            zone_price = last_row['High']
            strength = "High"

    if signal != "Neutral":
        return {
            'Symbol': last_row.get('Symbol', 'Unknown'),
            'Date': last_row['Date'],
            'Close': curr_price,
            'Trade_Signal': signal,
            'Zone_Price': zone_price,
            'Strength': strength
        }
    return None

# ==========================================
# MAIN EXECUTION
# ==========================================
def generate_signals():
    print("🚀 STARTING MARKET SCANNER...")
    init_signal_table()
    conn = get_db_connection()
    
    # 1. Get list of all unique symbols in your DB
    try:
        symbols = pd.read_sql("SELECT DISTINCT Symbol FROM cash_market_full", conn)['Symbol'].tolist()
    except Exception as e:
        print(f"❌ Error reading symbols: {e}")
        return

    print(f"📋 Scanning {len(symbols)} stocks for setups...")
    
    valid_signals = []
    
    for i, symbol in enumerate(symbols):
        # Fetch last 30 days of data for this stock
        query = f"SELECT * FROM cash_market_full WHERE Symbol='{symbol}' ORDER BY Date DESC LIMIT 30"
        df = pd.read_sql(query, conn)
        
        if not df.empty:
            # Sort ascending for calculations
            df = df.sort_values('Date').reset_index(drop=True)
            result = analyze_stock(df)
            
            if result:
                # Ensure Symbol is correct if missing from row
                result['Symbol'] = symbol 
                valid_signals.append(result)
                print(f"   found: {symbol} -> {result['Trade_Signal']}")

    # 2. Save to Database
    if valid_signals:
        print(f"\n💾 Saving {len(valid_signals)} potential trades to DB...")
        df_signals = pd.DataFrame(valid_signals)
        df_signals.to_sql('stocks_to_trade', conn, if_exists='append', index=False)
        print("✅ Signal Generation Complete.")
    else:
        print("⚠️ No strong signals found today.")

    conn.close()

if __name__ == "__main__":
    generate_signals()