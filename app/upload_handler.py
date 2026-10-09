import pandas as pd
import numpy as np
import os
import pickle

def process_uploaded_data(raw_df, col_mappings, base_dir):
    df = raw_df.copy()
    
    # Rename columns based on mapping
    rename_dict = {v: k for k, v in col_mappings.items()}
    df.rename(columns=rename_dict, inplace=True)
    
    # Drop unmatched columns to save memory
    df = df[list(col_mappings.keys())]
    
    # Cast to string for specific columns
    for col in ['invoice', 'stockcode', 'description', 'country']:
        df[col] = df[col].astype(str)
        
    # Coerce Dates and Numerics
    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Drop rows with unparseable required values
    invalid_mask = df['invoicedate'].isna() | df['quantity'].isna() | df['price'].isna()
    df = df[~invalid_mask].copy()
    
    # Drop exact duplicates
    df.drop_duplicates(inplace=True)
    
    # Business Rules
    df['is_cancellation'] = df['invoice'].str.upper().str.startswith('C')
    df['is_return'] = df['quantity'] < 0
    df['revenue'] = df['quantity'] * df['price']
    
    # Drop negative prices (bad data)
    df = df[df['price'] >= 0].copy()
    
    # Split datasets
    agg_trans = df.copy()
    cust_trans = df[df['customer_id'].notna()].copy()
    
    if len(cust_trans) == 0:
        raise ValueError("No valid customer IDs found in the uploaded data.")
        
    # Feature Engineering (RFM)
    obs_date = cust_trans['invoicedate'].max() + pd.Timedelta(days=1)
    valid_purchases = cust_trans[~cust_trans['is_return'] & ~cust_trans['is_cancellation']]
    
    rf = valid_purchases.groupby('customer_id').agg({
        'invoicedate': lambda x: (obs_date - x.max()).days,
        'invoice': 'nunique'
    }).reset_index()
    rf.rename(columns={'invoicedate': 'recency', 'invoice': 'frequency'}, inplace=True)
    
    m = cust_trans.groupby('customer_id')['revenue'].sum().reset_index()
    m.rename(columns={'revenue': 'monetary'}, inplace=True)
    
    rfm = pd.merge(rf, m, on='customer_id', how='outer')
    max_recency = rfm['recency'].max()
    rfm['recency'] = rfm['recency'].fillna(max_recency)
    rfm['frequency'] = rfm['frequency'].fillna(0)
    rfm['monetary'] = rfm['monetary'].fillna(0)
    rfm['aov'] = np.where(rfm['frequency'] > 0, rfm['monetary'] / rfm['frequency'], 0)
    
    # Load Models for inference
    models_dir = os.path.join(base_dir, 'models')
    
    # Segmentation
    clusters = pd.DataFrame()
    profiles = pd.DataFrame()
    try:
        with open(os.path.join(models_dir, 'rfm_scaler.pkl'), 'rb') as f:
            cluster_scaler = pickle.load(f)
        with open(os.path.join(models_dir, 'kmeans_model.pkl'), 'rb') as f:
            kmeans_model = pickle.load(f)
            
        features = ['recency', 'frequency', 'monetary']
        X_cluster = rfm[features]
        X_cluster_scaled = cluster_scaler.transform(X_cluster)
        rfm['cluster'] = kmeans_model.predict(X_cluster_scaled)
        
        clusters = rfm[['customer_id', 'cluster', 'recency', 'frequency', 'monetary']].copy()
        
        cluster_profiles = rfm.groupby('cluster')[['recency', 'frequency', 'monetary', 'aov']].median().reset_index()
        cluster_sizes = rfm['cluster'].value_counts().reset_index()
        cluster_sizes.columns = ['cluster', 'customer_count']
        profiles = pd.merge(cluster_sizes, cluster_profiles, on='cluster')
    except Exception as e:
        print(f"Clustering error: {e}")
        
    # Inactivity Prediction
    predictions = pd.DataFrame()
    try:
        with open(os.path.join(models_dir, 'inactivity_scaler.pkl'), 'rb') as f:
            inact_scaler = pickle.load(f)
        with open(os.path.join(models_dir, 'inactivity_model.pkl'), 'rb') as f:
            inact_model = pickle.load(f)
            
        X_pred = rfm[['recency', 'frequency', 'monetary', 'aov']]
        X_pred_scaled = inact_scaler.transform(X_pred)
        
        risk_scores = inact_model.predict_proba(X_pred_scaled)[:, 1]
        
        predictions = pd.DataFrame({
            'customer_id': rfm['customer_id'],
            'is_inactive': -1, # -1 means 'Unknown (Predicted Only)'
            'inactivity_risk_score': risk_scores
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        
    return {
        'agg_trans': agg_trans,
        'rfm': rfm,
        'clusters': clusters,
        'profiles': profiles,
        'predictions': predictions,
        'metrics': None # Uploaded data does not have a trained model's temporal metrics
    }
