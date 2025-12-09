import sqlite3
import pandas as pd
import nselib
from nselib import capital_market
from datetime import datetime, timedelta
import os
import time
import warnings

# ==========================================
# 1. CONFIGURATION
# ==========================================
warnings.simplefilter(action='ignore', category=FutureWarning)
DB_NAME = "market_data.db"
TABLE_NAME = "equal_market_full"

# ==========================================
# 2. DATABASE MANAGEMENT
# ==========================================
def init_db():
    """Creates the Cash Market Table if not exists."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Schema matching 'price_volume_and_deliverable_position_data'
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            Date TEXT,
            Symbol TEXT,
            Series TEXT,
            Prev_Close REAL,
            Open REAL,
            High REAL,
            Low REAL,
            Last REAL,
            Close REAL,
            Avg_Price REAL,
            Volume INTEGER,
            Turnover REAL,
            Trades INTEGER,
            Delivery_Qty INTEGER,
            Delivery_Pct REAL,
            PRIMARY KEY (Date, Symbol, Series)
        )
    ''')
    conn.commit()
    conn.close()
    print(f"✅ Table '{TABLE_NAME}' ready in {DB_NAME}.")

def get_last_date(symbol):
    """Finds the last date present in DB for a specific stock."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(Date) FROM {TABLE_NAME} WHERE Symbol = ? AND Series = 'EQ'", (symbol,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def save_to_db(df):
    """Saves data to SQLite."""
    if df is None or df.empty: return
    
    conn = sqlite3.connect(DB_NAME)
    try:
        df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
    except sqlite3.IntegrityError:
        pass # Ignore duplicates
    except Exception as e:
        print(f"   ❌ DB Error: {e}")
    finally:
        conn.close()

# ==========================================
# 3. DATA CLEANING & FETCHING
# ==========================================
def clean_cash_data(df, symbol):
    if df is None or df.empty: return None

    # 1. Map NSE Columns to DB Columns
    col_map = {
        'Date': 'Date',
        'Symbol': 'Symbol',
        'Series': 'Series',
        'PrevClose': 'Prev_Close',
        'OpenPrice': 'Open',
        'HighPrice': 'High',
        'LowPrice': 'Low',
        'LastPrice': 'Last',
        'ClosePrice': 'Close',
        'AveragePrice': 'Avg_Price',
        'TotalTradedQuantity': 'Volume',
        'TurnoverInRs': 'Turnover',
        'No.ofTrades': 'Trades',
        'DeliverableQty': 'Delivery_Qty',
        '%DlyQttoTradedQty': 'Delivery_Pct'
    }
    df = df.rename(columns=col_map)

    # 2. Filter for EQ Series only (Actual Equity)
    if 'Series' in df.columns:
        df = df[df['Series'] == 'EQ']

    # 3. Clean Numeric (Remove commas)
    numeric_cols = ['Prev_Close', 'Open', 'High', 'Low', 'Last', 'Close', 'Avg_Price', 
                    'Volume', 'Turnover', 'Trades', 'Delivery_Qty', 'Delivery_Pct']
    
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        else:
            df[col] = 0 # Fill missing numeric cols with 0

    # 4. Parse Dates
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    except:
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

    # Format for DB (YYYY-MM-DD)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['Symbol'] = symbol

    # Final Column Selection
    final_cols = ['Date', 'Symbol', 'Series', 'Prev_Close', 'Open', 'High', 'Low', 'Last', 
                  'Close', 'Avg_Price', 'Volume', 'Turnover', 'Trades', 'Delivery_Qty', 'Delivery_Pct']
    
    # Ensure columns exist
    for c in final_cols:
        if c not in df.columns: df[c] = None
            
    return df[final_cols].reset_index(drop=True)

def sync_stock(symbol):
    last_date_str = get_last_date(symbol)
    
    try:
        if last_date_str is None:
            # 1. New Stock: Get 1 Year History
            # print(f"   🆕 {symbol}: Fetching 1 Year Cash Data...")
            # Note: capital_market API uses period='1y' or from_date/to_date
            df = capital_market.price_volume_and_deliverable_position_data(symbol=symbol, period='1y')
            cleaned = clean_cash_data(df, symbol)
            save_to_db(cleaned)
            
        else:
            # 2. Existing Stock: Incremental Sync
            last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
            today = datetime.now()
            
            if last_date.date() >= today.date():
                return # Already up to date
                
            start_date = last_date + timedelta(days=1)
            
            # FIX: Handle single day fetch (nselib validation error)
            req_to_date = today
            if start_date.date() == today.date():
                req_to_date = today + timedelta(days=1)
            
            # print(f"   🔄 {symbol}: Syncing from {start_date.date()}...")
            
            df = capital_market.price_volume_and_deliverable_position_data(
                symbol=symbol, 
                from_date=start_date.strftime('%d-%m-%Y'), 
                to_date=req_to_date.strftime('%d-%m-%Y')
            )
            cleaned = clean_cash_data(df, symbol)
            save_to_db(cleaned)
            
    except Exception as e:
        # print(f"   ⚠️ Failed {symbol}: {e}")
        pass

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    init_db()
    
    print("🚀 FETCHING MASTER STOCK LIST (F&O)...")
    try:
        # Get list of all F&O Stocks
        fno_df = capital_market.equity_list()
        all_symbols = fno_df['SYMBOL'].tolist()
        
        # Filter valid strings
        all_symbols = [s for s in all_symbols if isinstance(s, str)]
        
        print(f"✅ Found {len(all_symbols)} Stocks.")
        print("⏳ Starting Batch Ingestion...")
        
        count = 0
        total = len(all_symbols)
        
        for sym in all_symbols:
            count += 1
            print(f"[{count}/{total}] Checking {sym}...", end="\r")
            sync_stock(sym)
            time.sleep(0.2) # Rate limit protection
            
        print("\n\n🎉 ALL CASH MARKET DATA SYNCED!")
        
    except Exception as e:
        print(f"\n❌ Error getting stock list: {e}")