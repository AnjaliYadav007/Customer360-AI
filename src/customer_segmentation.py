import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def segment_customers(processed_dir, models_dir):
    print("Starting Customer Segmentation...")
    
    input_path = os.path.join(processed_dir, 'customer_rfm.parquet')
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        sys.exit(1)
        
    df = pd.read_parquet(input_path)
    print(f"Loaded RFM features for {len(df)} customers.")
    
    # 1. Preprocessing and Scaling
    # RFM metrics are highly right-skewed. Monetary can be negative due to returns.
    # Yeo-Johnson transformation naturally handles zero and negative values while fixing skewness.
    features = ['recency', 'frequency', 'monetary']
    X = df[features]
    
    scaler = PowerTransformer(method='yeo-johnson')
    X_scaled = scaler.fit_transform(X)
    
    # 2. Determine Optimal k using Silhouette Score
    # We will test k in [2, 3, 4, 5] and select the one with the highest silhouette score.
    best_k = 3
    best_score = -1
    best_model = None
    
    print("Evaluating clusters (k=2 to 5)...")
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f" - k={k}, Silhouette Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans
            
    print(f"Selected Optimal Clusters: {best_k} (Silhouette Score: {best_score:.4f})")
    
    # Assign labels to dataframe
    df['cluster'] = best_model.labels_
    
    # 3. Profiling
    # Map clusters to readable segments based on their median RFM values
    cluster_profiles = df.groupby('cluster')[['recency', 'frequency', 'monetary', 'aov']].median().reset_index()
    cluster_sizes = df['cluster'].value_counts().reset_index()
    cluster_sizes.columns = ['cluster', 'customer_count']
    cluster_profiles = pd.merge(cluster_sizes, cluster_profiles, on='cluster')
    
    print("\n--- Cluster Profiles (Medians) ---")
    print(cluster_profiles)
    
    # 4. Save Artifacts
    os.makedirs(models_dir, exist_ok=True)
    
    # Save Model & Scaler
    with open(os.path.join(models_dir, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(models_dir, 'rfm_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    # Save Customer Assignments
    assignments_path = os.path.join(processed_dir, 'customer_clusters.parquet')
    df.to_parquet(assignments_path, index=False)
    
    # Save Profiles
    profiles_path = os.path.join(processed_dir, 'cluster_profiles.csv')
    cluster_profiles.to_csv(profiles_path, index=False)
    
    print(f"\nSaved models to {models_dir}")
    print(f"Saved assignments to {assignments_path}")
    print(f"Saved profiles to {profiles_path}")
    
    print("\n--- Important Limitation Note ---")
    print("This RFM segmentation is a point-in-time snapshot derived from the entire historical dataset.")
    print("It provides a descriptive view of customer behavior but does NOT constitute a time-based validation of future behavior.")
    print("For predictive modeling (e.g., predicting future churn or inactivity), the dataset must be properly split using temporal cross-validation, ensuring features are calculated strictly on data prior to the observation window.")

if __name__ == "__main__":
    processed = r"data\processed"
    models = r"models"
    segment_customers(processed, models)
