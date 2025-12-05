import sqlite3
import pandas as pd
import nselib
from nselib import derivatives
from datetime import datetime, timedelta
import os
import warnings

# ==========================================
# 1. CONFIGURATION
# ==========================================
warnings.simplefilter(action='ignore', category=FutureWarning)
DB_NAME = "market_data.db"

# Major F&O Indices to track (You can add stocks here too)
ALL_SYMBOLS = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# ==========================================
# 2. DATABASE MANAGEMENT
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 🆕 UPDATED SCHEMA: Includes ALL NSE Fields
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS index_derivative (
            Date TEXT,
            Instrument TEXT,
            Symbol TEXT,
            Expiry TEXT,
            Strike REAL,
            Option_Type TEXT,
            Market_Type TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Last REAL,
            Prev_Close REAL,
            Settle_Price REAL,
            Volume INTEGER,
            Turnover REAL,
            OI INTEGER,
            Change_OI INTEGER,
            Lot_Size INTEGER,
            Spot REAL,
            PRIMARY KEY (Date, Symbol, Instrument)
        )
    ''')
    conn.commit()
    conn.close()
    print(f"✅ Database {DB_NAME} ready with FULL Schema.")

def get_last_date(symbol):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Check max date for this symbol
    cursor.execute("SELECT MAX(Date) FROM index_derivative WHERE Symbol = ?", (symbol,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

def save_to_db(df):
    if df is None or df.empty: return
    conn = sqlite3.connect(DB_NAME)
    try:
        df.to_sql('index_derivative', conn, if_exists='append', index=False)
        print(f"   💾 Saved {len(df)} rows to DB.")
    except sqlite3.IntegrityError:
        print("   ⚠️  Data overlap/duplicate detected. Ignored.")
    except Exception as e:
        print(f"   ❌ Database Error: {e}")
    finally:
        conn.close()

# ==========================================
# 3. DATA CLEANING (ALL FIELDS)
# ==========================================
def clean_nse_data(df, symbol):
    if df is None or df.empty: return None

    # 1. Map ALL nselib columns to DB columns
    col_map = {
        'TIMESTAMP': 'Date',
        'INSTRUMENT': 'Instrument',
        'SYMBOL': 'Symbol',
        'EXPIRY_DT': 'Expiry',
        'STRIKE_PRICE': 'Strike',
        'OPTION_TYPE': 'Option_Type',
        'MARKET_TYPE': 'Market_Type',
        'OPENING_PRICE': 'Open',
        'TRADE_HIGH_PRICE': 'High',
        'TRADE_LOW_PRICE': 'Low',
        'CLOSING_PRICE': 'Close',
        'LAST_TRADED_PRICE': 'Last',
        'PREV_CLS': 'Prev_Close',
        'SETTLE_PRICE': 'Settle_Price',
        'TOT_TRADED_QTY': 'Volume',
        'TOT_TRADED_VAL': 'Turnover',
        'OPEN_INT': 'OI',
        'CHANGE_IN_OI': 'Change_OI',
        'MARKET_LOT': 'Lot_Size',
        'UNDERLYING_VALUE': 'Spot'
    }
    df = df.rename(columns=col_map)

    # 2. Clean Numeric Columns (Remove commas)
    numeric_cols = [
        'Strike', 'Open', 'High', 'Low', 'Close', 'Last', 'Prev_Close', 
        'Settle_Price', 'Volume', 'Turnover', 'OI', 'Change_OI', 'Lot_Size', 'Spot'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        else:
            df[col] = 0 # Fill missing numeric cols

    # 3. Parse Dates
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
        df['Expiry'] = pd.to_datetime(df['Expiry'], format='%d-%b-%Y')
    except:
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        df['Expiry'] = pd.to_datetime(df['Expiry'], infer_datetime_format=True)

    # 🛑 Continuous Contract Logic: 
    # Group by Date and keep ONLY the row with the Nearest Expiry (Near Month Future)
    df = df.sort_values(['Date', 'Expiry'])
    df = df.loc[df.groupby('Date')['Expiry'].idxmin()]

    # 4. Format Dates for SQLite
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['Expiry'] = df['Expiry'].dt.strftime('%Y-%m-%d')
    
    # Ensure Symbol is set correctly
    df['Symbol'] = symbol
    
    # 5. Select Final Ordered Columns matching DB
    final_cols = [
        'Date', 'Instrument', 'Symbol', 'Expiry', 'Strike', 'Option_Type', 'Market_Type',
        'Open', 'High', 'Low', 'Close', 'Last', 'Prev_Close', 'Settle_Price',
        'Volume', 'Turnover', 'OI', 'Change_OI', 'Lot_Size', 'Spot'
    ]
    
    # Ensure all columns exist
    for c in final_cols:
        if c not in df.columns: df[c] = None

    return df[final_cols].reset_index(drop=True)

# ==========================================
# 4. SYNC LOGIC
# ==========================================
def sync_data(symbol):
    print(f"\n🔄 Checking Data for {symbol}...")
    
    last_date_str = get_last_date(symbol)
    today = datetime.now()
    
    if last_date_str is None:
        print(f"   🆕 Data not in DB. Fetching full 1 Year history...")
        try:
            # Fetch 1 Year Data
            df = derivatives.future_price_volume_data(symbol=symbol, instrument='FUTIDX', period='1W')
            cleaned_df = clean_nse_data(df, symbol)
            save_to_db(cleaned_df)
        except Exception as e:
            print(f"   ❌ Error fetching 1W data: {e}")
            
    else:
        # Incremental Sync
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
        
        if last_date.date() >= today.date():
            print("   ✅ Data is already up to date.")
            return

        start_date = last_date + timedelta(days=1)
        
        # FIX: If start_date is today, nselib complains if from_date == to_date.
        # Workaround: Set to_date to tomorrow so the range is valid.
        req_to_date = today
        if start_date.date() == today.date():
            req_to_date = today + timedelta(days=1)
            
        print(f"   📅 DB has data up to {last_date_str}. Fetching from {start_date.date()}...")
        
        try:
            df = derivatives.future_price_volume_data(
                symbol=symbol, 
                instrument='FUTIDX', 
                from_date=start_date.strftime('%d-%m-%Y'), 
                to_date=req_to_date.strftime('%d-%m-%Y')
            )
            cleaned_df = clean_nse_data(df, symbol)
            save_to_db(cleaned_df)
        except Exception as e:
            print(f"   ❌ Error syncing: {e}")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    init_db()
    
    print(f"🚀 Starting Ingestion for: {ALL_SYMBOLS}")
    
    for sym in ALL_SYMBOLS:
        sync_data(sym)
        
    print("\n✅ All Data Synced.")