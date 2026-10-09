import pandas as pd
import numpy as np
import os
import sys

def preprocess_data(raw_path, processed_dir):
    print("Starting data preprocessing pipeline...")
    
    # 1. Load Data
    try:
        df = pd.concat(pd.read_excel(raw_path, engine="calamine", sheet_name=None), ignore_index=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        
    initial_rows = len(df)
    print(f"Initial rows loaded: {initial_rows}")

    # 2. Standardize column names
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    required_cols = {'invoice', 'stockcode', 'description', 'quantity', 'invoicedate', 'price', 'customer_id', 'country'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"Error: Missing required columns {missing_cols}")
        sys.exit(1)
        
    # Cast mixed-type columns to string for Parquet compatibility
    for col in ['invoice', 'stockcode', 'description', 'country']:
        df[col] = df[col].astype(str)
        
    # 3. Parse Dates and Numerics safely
    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    invalid_dates = df['invoicedate'].isna().sum()
    invalid_qty = df['quantity'].isna().sum()
    invalid_price = df['price'].isna().sum()
    print(f"Invalid dates (coerced to NaT): {invalid_dates}")
    print(f"Invalid quantities (coerced to NaN): {invalid_qty}")
    print(f"Invalid prices (coerced to NaN): {invalid_price}")
    
    # Drop rows with invalid required numerics/dates that couldn't be parsed
    invalid_mask = df['invoicedate'].isna() | df['quantity'].isna() | df['price'].isna()
    df = df[~invalid_mask].copy()
    
    # 4. Detect and remove duplicates
    duplicates_count = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"Duplicates removed: {duplicates_count}")
    
    # 5. Identify Cancellations and Returns
    # In UCI dataset, Invoice starting with 'C' indicates a cancellation.
    # Also look for negative quantities and prices.
    df['is_cancellation'] = df['invoice'].astype(str).str.upper().str.startswith('C')
    df['is_return'] = df['quantity'] < 0
    df['revenue'] = df['quantity'] * df['price']
    
    cancellations_count = df['is_cancellation'].sum()
    negative_qty = df['is_return'].sum()
    negative_price = (df['price'] < 0).sum()
    zero_price = (df['price'] == 0).sum()
    
    print(f"Cancellations (Invoice starts with 'C'): {cancellations_count}")
    print(f"Returns (Negative quantities): {negative_qty}")
    print(f"Negative prices (bad debt/errors): {negative_price}")
    print(f"Zero prices (free items/errors): {zero_price}")
    
    # Handle negative prices (usually bad data, we will drop them)
    df = df[df['price'] >= 0].copy()
    
    # 6. Separate missing Customer IDs
    missing_cust_mask = df['customer_id'].isna()
    missing_cust_count = missing_cust_mask.sum()
    print(f"Missing Customer IDs: {missing_cust_count}")
    
    # Aggregate dataset (includes missing customer IDs, returns, everything valid)
    df_aggregate = df.copy()
    
    # Customer-level dataset (excludes missing customer IDs)
    df_customer = df[~missing_cust_mask].copy()
    
    # 7. Save to Processed
    os.makedirs(processed_dir, exist_ok=True)
    
    # Determine output format
    try:
        import pyarrow
        fmt = 'parquet'
    except ImportError:
        fmt = 'csv'
        
    agg_path = os.path.join(processed_dir, f'aggregate_transactions.{fmt}')
    cust_path = os.path.join(processed_dir, f'customer_transactions.{fmt}')
    
    if fmt == 'parquet':
        df_aggregate.to_parquet(agg_path, index=False)
        df_customer.to_parquet(cust_path, index=False)
    else:
        df_aggregate.to_csv(agg_path, index=False)
        df_customer.to_csv(cust_path, index=False)
        
    print(f"\n--- Data Quality Report ---")
    print(f"Input Rows: {initial_rows}")
    print(f"Duplicates Removed: {duplicates_count}")
    print(f"Invalid Rows Dropped: {invalid_mask.sum()} (dates/numerics)")
    print(f"Negative Price Rows Dropped: {negative_price}")
    print(f"Missing Customer IDs: {missing_cust_count}")
    print(f"Cancellations/Returns Flagged: {cancellations_count}")
    print(f"Retained Aggregate Rows: {len(df_aggregate)}")
    print(f"Retained Customer Rows: {len(df_customer)}")
    print(f"Format used: {fmt}")
    print(f"Outputs:\n  - {agg_path}\n  - {cust_path}")

if __name__ == "__main__":
    raw = r"data\raw\online_retail_II.xlsx"
    processed = r"data\processed"
    preprocess_data(raw, processed)
