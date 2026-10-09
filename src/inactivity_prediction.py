import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb

def predict_inactivity(processed_dir, models_dir):
    print("Starting Inactivity Prediction Modeling...")
    
    input_path = os.path.join(processed_dir, 'customer_transactions.parquet')
    df = pd.read_parquet(input_path)
    
    # 1. Temporal Split Definition
    # We use the last 90 days of the dataset as the Future Outcome Window.
    # The rest is the Historical Observation Window.
    max_date = df['invoicedate'].max()
    cutoff_date = max_date - pd.Timedelta(days=90)
    print(f"Dataset Max Date: {max_date}")
    print(f"Cutoff Date (Observation vs Future): {cutoff_date}")
    
    # 2. Build Temporal Cohort
    # Observation data (Features)
    obs_df = df[df['invoicedate'] < cutoff_date].copy()
    # Future data (Target)
    future_df = df[df['invoicedate'] >= cutoff_date].copy()
    
    # We only predict for customers who were "alive" (made at least 1 purchase) in the observation window
    valid_purchases_obs = obs_df[~obs_df['is_return'] & ~obs_df['is_cancellation']]
    eligible_customers = valid_purchases_obs['customer_id'].unique()
    print(f"Eligible customers (active in observation window): {len(eligible_customers)}")
    
    # 3. Build Predictive Features (Strictly from observation window)
    rf = valid_purchases_obs.groupby('customer_id').agg({
        'invoicedate': lambda x: (cutoff_date - x.max()).days,
        'invoice': 'nunique'
    }).reset_index()
    rf.rename(columns={'invoicedate': 'recency', 'invoice': 'frequency'}, inplace=True)
    
    m = obs_df.groupby('customer_id')['revenue'].sum().reset_index()
    m.rename(columns={'revenue': 'monetary'}, inplace=True)
    
    features = pd.merge(rf, m, on='customer_id', how='left')
    features['aov'] = np.where(features['frequency'] > 0, features['monetary'] / features['frequency'], 0)
    
    # Ensure all eligible customers are in features
    assert len(features) == len(eligible_customers), "Feature count mismatch!"
    
    # 4. Define Target Label
    # Inactive = 1 if 0 valid purchases in Future Window, 0 otherwise
    valid_purchases_future = future_df[~future_df['is_return'] & ~future_df['is_cancellation']]
    active_future_customers = valid_purchases_future['customer_id'].unique()
    
    features['is_inactive'] = (~features['customer_id'].isin(active_future_customers)).astype(int)
    inactive_rate = features['is_inactive'].mean()
    print(f"Inactivity Rate (Churn): {inactive_rate:.2%}")
    
    # 5. Prepare Modeling
    X = features[['recency', 'frequency', 'monetary', 'aov']]
    y = features['is_inactive']
    
    # Train-test split on the cohort (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train Models
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), random_state=42, eval_metric='logloss')
    }
    
    results = {}
    best_model_name = None
    best_auc = 0
    
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        if name in ['LogisticRegression', 'RandomForest']:
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train) # XGBoost handles unscaled data well
            probs = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)
            
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        
        print(f"\nModel: {name}")
        print(f"ROC-AUC: {auc:.4f}")
        print("Confusion Matrix:\n", cm)
        
        results[name] = {
            'roc_auc': float(auc),
            'confusion_matrix': cm.tolist()
        }
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with ROC-AUC {best_auc:.4f}")
    
    # 7. Generate Predictions using Best Model for all eligible customers
    best_model = models[best_model_name]
    if best_model_name in ['LogisticRegression', 'RandomForest']:
        X_all_scaled = scaler.transform(X)
        risk_scores = best_model.predict_proba(X_all_scaled)[:, 1]
    else:
        risk_scores = best_model.predict_proba(X)[:, 1]
        
    features['inactivity_risk_score'] = risk_scores
    
    # 8. Save Artifacts
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'inactivity_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(models_dir, 'inactivity_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(models_dir, 'inactivity_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    predictions_path = os.path.join(processed_dir, 'inactivity_predictions.parquet')
    features[['customer_id', 'is_inactive', 'inactivity_risk_score']].to_parquet(predictions_path, index=False)
    print(f"\nSaved models to {models_dir}")
    print(f"Saved predictions to {predictions_path}")
    
if __name__ == "__main__":
    processed = r"data\processed"
    models = r"models"
    predict_inactivity(processed, models)
