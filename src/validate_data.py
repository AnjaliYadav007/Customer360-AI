import pandas as pd
import os
import sys

def validate_outputs(processed_dir):
    print("Starting validation...")
    
    fmt = 'parquet' if os.path.exists(os.path.join(processed_dir, 'aggregate_transactions.parquet')) else 'csv'
    
    agg_path = os.path.join(processed_dir, f'aggregate_transactions.{fmt}')
    cust_path = os.path.join(processed_dir, f'customer_transactions.{fmt}')
    
    if not os.path.exists(agg_path) or not os.path.exists(cust_path):
        print("Error: Processed files not found!")
        sys.exit(1)
        
    print(f"Reading {fmt} files...")
    if fmt == 'parquet':
        df_agg = pd.read_parquet(agg_path)
        df_cust = pd.read_parquet(cust_path)
    else:
        df_agg = pd.read_csv(agg_path, parse_dates=['invoicedate'])
        df_cust = pd.read_csv(cust_path, parse_dates=['invoicedate'])
        
    # 1. Required Columns
    required_cols = {'invoice', 'stockcode', 'description', 'quantity', 'invoicedate', 'price', 'customer_id', 'country', 'is_cancellation', 'is_return', 'revenue'}
    assert required_cols.issubset(set(df_agg.columns)), "Aggregate dataset missing columns!"
    assert required_cols.issubset(set(df_cust.columns)), "Customer dataset missing columns!"
    print("[OK] Required columns validated.")
    
    # 2. Date Parsing
    assert pd.api.types.is_datetime64_any_dtype(df_agg['invoicedate']), "Aggregate invoice date not parsed as datetime!"
    assert pd.api.types.is_datetime64_any_dtype(df_cust['invoicedate']), "Customer invoice date not parsed as datetime!"
    print("[OK] Date parsing validated.")
    
    # 3. Customer ID Handling
    assert df_agg['customer_id'].isna().sum() > 0, "Aggregate dataset should retain missing customer IDs!"
    assert df_cust['customer_id'].isna().sum() == 0, "Customer dataset should NOT have missing customer IDs!"
    print("[OK] Customer ID separation validated.")
    
    # 4. Numerics Valid
    assert df_agg['price'].min() >= 0, "Aggregate dataset should not have negative prices!"
    assert df_cust['price'].min() >= 0, "Customer dataset should not have negative prices!"
    print("[OK] Numerics validated.")
    
    # 5. Row Count Reconciliation
    missing_cust_count = df_agg['customer_id'].isna().sum()
    assert len(df_agg) == len(df_cust) + missing_cust_count, "Row counts do not reconcile between aggregate and customer datasets!"
    print("[OK] Row count reconciliation validated.")
    
    # 6. Business Rules (Flags and Revenue)
    assert (df_agg['is_return'] == (df_agg['quantity'] < 0)).all(), "is_return flag does not strictly match quantity < 0!"
    
    # Check that revenue is correctly calculated (accounting for floating point imprecision if any, but since we are doing direct mult it should match exactly here or close enough, but wait let's use np.isclose if needed. Actually exact should work)
    assert (df_agg['revenue'].round(4) == (df_agg['quantity'] * df_agg['price']).round(4)).all(), "Revenue calculation is incorrect."
    print("[OK] Business rules (returns and revenue) validated.")
    
    # 7. Validate Feature Engineering Output (RFM)
    rfm_path = os.path.join(processed_dir, 'customer_rfm.parquet')
    if os.path.exists(rfm_path):
        df_rfm = pd.read_parquet(rfm_path)
        assert df_rfm['customer_id'].isna().sum() == 0, "Missing IDs in RFM table!"
        assert len(df_rfm) == df_rfm['customer_id'].nunique(), "Duplicate customers in RFM table!"
        assert len(df_rfm) == len(df_cust['customer_id'].unique()), "RFM customer count doesn't match processed valid customers!"
        assert (df_rfm['recency'] >= 0).all(), "Recency cannot be negative!"
        assert (df_rfm['frequency'] >= 0).all(), "Frequency cannot be negative!"
        print("[OK] RFM Features validated.")
        
    # 8. Validate Segmentation Output
    cluster_path = os.path.join(processed_dir, 'customer_clusters.parquet')
    if os.path.exists(cluster_path):
        df_clusters = pd.read_parquet(cluster_path)
        assert len(df_clusters) == len(df_rfm), "Cluster assignment count mismatch!"
        assert 'cluster' in df_clusters.columns, "Cluster assignments missing!"
        print("[OK] Customer clusters validated.")
        
    # 9. Validate Inactivity Prediction Output
    pred_path = os.path.join(processed_dir, 'inactivity_predictions.parquet')
    if os.path.exists(pred_path):
        df_pred = pd.read_parquet(pred_path)
        assert 'customer_id' in df_pred.columns, "Missing customer_id in predictions."
        assert 'is_inactive' in df_pred.columns, "Missing is_inactive target."
        assert 'inactivity_risk_score' in df_pred.columns, "Missing risk scores."
        assert df_pred['is_inactive'].isin([0, 1]).all(), "Target label must be 0 or 1."
        assert (df_pred['inactivity_risk_score'] >= 0).all() and (df_pred['inactivity_risk_score'] <= 1).all(), "Risk scores must be probabilities between 0 and 1."
        assert df_pred['customer_id'].duplicated().sum() == 0, "Duplicate customer_id in predictions."
        print("[OK] Inactivity predictions validated.")
        
    print("All validations passed successfully!")

if __name__ == "__main__":
    processed = r"data\processed"
    validate_outputs(processed)
