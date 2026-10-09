import pandas as pd
import numpy as np
import os
import sys

def engineer_features(processed_dir):
    print("Starting Feature Engineering...")
    
    input_path = os.path.join(processed_dir, 'customer_transactions.parquet')
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        sys.exit(1)
        
    df = pd.read_parquet(input_path)
    
    # Standardize column names just in case
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # 1. Observation Date
    # We derive the observation date as the max date in the dataset + 1 day
    # This prevents the most recent purchase from having Recency = 0
    obs_date = df['invoicedate'].max() + pd.Timedelta(days=1)
    print(f"Observation Date set to: {obs_date}")
    
    # 2. Separate valid purchases from returns
    # We want Frequency and Recency to only count true purchase events
    valid_purchases = df[~df['is_return'] & ~df['is_cancellation']].copy()
    
    # 3. Calculate RFM components
    
    # Recency: days since most recent valid purchase
    # Frequency: count of unique valid invoices
    rf = valid_purchases.groupby('customer_id').agg({
        'invoicedate': lambda x: (obs_date - x.max()).days,
        'invoice': 'nunique'
    }).reset_index()
    rf.rename(columns={'invoicedate': 'recency', 'invoice': 'frequency'}, inplace=True)
    
    # Monetary: net revenue per customer
    # We use the FULL customer dataset to allow negative quantities (returns) to correctly offset revenue
    m = df.groupby('customer_id')['revenue'].sum().reset_index()
    m.rename(columns={'revenue': 'monetary'}, inplace=True)
    
    # Additional features: Average Order Value (AOV)
    # AOV = Monetary / Frequency (handle division by zero if freq=0)
    
    # Merge RFM
    rfm = pd.merge(rf, m, on='customer_id', how='outer')
    
    # For customers who only had returns (Frequency = NaN or 0), 
    # they shouldn't realistically be considered active purchasers for this window.
    # We will fill missing recency with a large number (e.g. max recency) and freq with 0.
    max_recency = rfm['recency'].max()
    rfm['recency'] = rfm['recency'].fillna(max_recency)
    rfm['frequency'] = rfm['frequency'].fillna(0)
    rfm['monetary'] = rfm['monetary'].fillna(0)
    
    rfm['aov'] = np.where(rfm['frequency'] > 0, rfm['monetary'] / rfm['frequency'], 0)
    
    print(f"RFM feature table created for {len(rfm)} customers.")
    
    # Output paths
    output_path = os.path.join(processed_dir, 'customer_rfm.parquet')
    rfm.to_parquet(output_path, index=False)
    print(f"Saved RFM features to {output_path}")

if __name__ == "__main__":
    processed = r"data\processed"
    engineer_features(processed)
