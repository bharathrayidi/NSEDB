import sqlite3
import pandas as pd
import numpy as np
from datetime import timedelta

# ==========================================
# CONFIGURATION
# ==========================================
DB_NAME = "market_data.db"
SOURCE_TABLE = "cash_market_full"
OUTPUT_TABLE = "volume_predictions"
VOLUME_MULTIPLIER = 3.0  # Look for volume 3x higher than average
LOOKBACK_WINDOW = 20     # Compare today's volume to the last 20 days average

# ==========================================
# 1. LOAD DATA
# ==========================================
conn = sqlite3.connect(DB_NAME)
print(f"⏳ Loading data from {SOURCE_TABLE}...")

try:
    # Load entire history. Ensure columns match your DB schema (Close, Volume, Symbol, Date)
    df = pd.read_sql(f"SELECT Date, Symbol, Open, High, Low, Close, Volume FROM {SOURCE_TABLE}", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort for calculations
    df = df.sort_values(by=['Symbol', 'Date'])
    
    # Handle zero volume (trading holidays or errors)
    df = df[df['Volume'] > 0]
    
except Exception as e:
    print(f"❌ Error reading database: {e}")
    exit()

print(f"✅ Loaded {len(df):,} rows. Processing strategy...")

# ==========================================
# 2. IDENTIFY VOLUME SPIKES (The "Trigger")
# ==========================================
# Group by Symbol to calculate indicators per stock
df['Prev_Volume_Avg'] = df.groupby('Symbol')['Volume'].transform(
    lambda x: x.rolling(window=LOOKBACK_WINDOW).mean().shift(1)
)

# Calculate the Volume Ratio (Today / Average of last 20 days)
df['Vol_Ratio'] = df['Volume'] / df['Prev_Volume_Avg']

# The Trigger: Volume is > 3x (or 5x) the average
df['Is_Triggered'] = df['Vol_Ratio'] >= VOLUME_MULTIPLIER

# ==========================================
# 3. PREDICT NEXT DAY (The "Outcome")
# ==========================================
# We want to see what happens the DAY AFTER the trigger
# Shift(-1) gets tomorrow's data onto today's row for comparison
df['Next_Open'] = df.groupby('Symbol')['Open'].shift(-1)
df['Next_Close'] = df.groupby('Symbol')['Close'].shift(-1)
df['Next_High'] = df.groupby('Symbol')['High'].shift(-1)
df['Next_Date'] = df.groupby('Symbol')['Date'].shift(-1)

# Calculate the actual move the next day
df['Next_Move_Pct'] = ((df['Next_Close'] - df['Next_Close'].shift(1)) / df['Next_Close'].shift(1)) * 100

# Simple Prediction Logic: 
# If Volume Spike + Price Green today -> Predict UP tomorrow
# If Volume Spike + Price Red today -> Predict DOWN tomorrow
df['Price_Change_Today'] = df['Close'] - df['Open']
df['Prediction'] = np.where(df['Price_Change_Today'] > 0, "UP", "DOWN")

# ==========================================
# 4. FILTER TRIGGERS & ANALYZE FAILURE
# ==========================================
# Keep only rows where the strategy triggered
triggers = df[df['Is_Triggered'] == True].copy()

# Determine Success/Failure
# Success = Predicted UP and Next Move > 0, OR Predicted DOWN and Next Move < 0
conditions = [
    (triggers['Prediction'] == "UP") & (triggers['Next_Move_Pct'] > 0),
    (triggers['Prediction'] == "DOWN") & (triggers['Next_Move_Pct'] < 0)
]
choices = ["Success", "Success"]
triggers['Outcome'] = np.select(conditions, choices, default="Fail")

# Why did it fail? (Simple analysis)
triggers['Failure_Reason'] = np.where(
    triggers['Outcome'] == "Fail",
    "Market Reversal", # Placeholder logic - can be enhanced with Nifty direction
    ""
)

# Format for Output
final_output = triggers[[
    'Date', 'Symbol', 'Close', 'Volume', 'Vol_Ratio', 
    'Prediction', 'Next_Date', 'Next_Close', 'Next_Move_Pct', 'Outcome', 'Failure_Reason'
]].dropna()

# Rename columns for clarity in DB
final_output.columns = [
    'Trigger_Date', 'Symbol', 'Trigger_Price', 'Trigger_Vol', 'Vol_Multiplier',
    'Predicted_Dir', 'Target_Date', 'Actual_Close', 'Actual_Move_Pct', 'Result', 'Failure_Reason'
]

# ==========================================
# 5. INGEST TO DATABASE
# ==========================================
print(f"📊 Found {len(final_output):,} triggers.")
print(f"💾 Saving to table: {OUTPUT_TABLE}...")

try:
    final_output.to_sql(OUTPUT_TABLE, conn, if_exists='replace', index=False)
    print("✅ Success! Data ingested.")
    
    # Show summary stats
    success_rate = (final_output['Result'] == "Success").mean()
    print(f"\n📈 Strategy Performance:")
    print(f"   Win Rate: {success_rate:.2%}")
    print(f"   Avg Return per Trade: {final_output['Actual_Move_Pct'].mean():.2f}%")
    
    print("\nSample Data:")
    print(final_output.tail(5))
    
except Exception as e:
    print(f"❌ Error saving to DB: {e}")

conn.close()