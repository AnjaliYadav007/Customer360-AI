import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json

from upload_handler import process_uploaded_data

# --- Config & Theming ---
st.set_page_config(page_title="Customer360 AI", layout="wide", initial_sidebar_state="expanded")

# Colors
BURGUNDY = "#6B1C2C"
GOLD = "#C9A84C"
CREAM = "#FEFCF8"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {CREAM};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {BURGUNDY};
    }}
    .stButton>button {{
        background-color: {GOLD} !important;
        color: {BURGUNDY} !important;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }}
    .stSidebar {{
        background-color: #F8F5F0;
    }}
</style>
""", unsafe_allow_html=True)

# --- Data Loading (Cached) ---
@st.cache_data
def load_demo_data():
    data = {}
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    
    try:
        data['agg_trans'] = pd.read_parquet(os.path.join(proc_dir, 'aggregate_transactions.parquet'))
        data['rfm'] = pd.read_parquet(os.path.join(proc_dir, 'customer_rfm.parquet'))
        data['clusters'] = pd.read_parquet(os.path.join(proc_dir, 'customer_clusters.parquet'))
        data['profiles'] = pd.read_csv(os.path.join(proc_dir, 'cluster_profiles.csv'))
        
        pred_path = os.path.join(proc_dir, 'inactivity_predictions.parquet')
        if os.path.exists(pred_path):
            data['predictions'] = pd.read_parquet(pred_path)
            
        metrics_path = os.path.join(models_dir, 'inactivity_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                data['metrics'] = json.load(f)
                
    except Exception as e:
        st.error(f"Error loading demo data: {e}")
    return data

def get_current_data():
    if 'uploaded_processed_data' in st.session_state and st.session_state.uploaded_processed_data is not None:
        return st.session_state.uploaded_processed_data
    return load_demo_data()

# --- Main App ---
def main():
    st.sidebar.title("Customer360 AI")
    st.sidebar.markdown("**Advanced Customer Intelligence**")
    
    if 'uploaded_processed_data' in st.session_state and st.session_state.uploaded_processed_data is not None:
        st.sidebar.success("Using Uploaded Data")
    else:
        st.sidebar.info("Using Default Demo Data")
    
    menu = ["Upload Data", "Business Overview", "Customer Segmentation", "Inactivity Prediction", "Customer Explorer", "Business Recommendations"]
    choice = st.sidebar.radio("Navigation", menu)
    
    if choice == "Upload Data":
        show_upload()
        return
        
    data = get_current_data()
    if not data or 'agg_trans' not in data:
        st.warning("Data not found. Please upload a dataset or run the demo preprocessing pipelines.")
        return
        
    if choice == "Business Overview":
        show_overview(data)
    elif choice == "Customer Segmentation":
        show_segmentation(data)
    elif choice == "Inactivity Prediction":
        show_inactivity(data)
    elif choice == "Customer Explorer":
        show_explorer(data)
    elif choice == "Business Recommendations":
        show_recommendations(data)

def show_upload():
    st.title("Upload Custom Data")
    st.markdown("Upload your own transaction dataset to run through the Customer360 AI pipeline in-memory.")
    
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state.raw_df = df
                st.session_state.last_uploaded = uploaded_file.name
                st.session_state.col_mappings = {} # Reset mappings
                st.success(f"Successfully loaded {uploaded_file.name} with {len(df):,} rows.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
                
        df = st.session_state.get('raw_df')
        if df is not None:
            st.subheader("Column Mapping")
            st.markdown("Ensure your columns map to the required fields. Our system attempts to auto-detect them.")
            expected_cols = ['invoice', 'stockcode', 'description', 'quantity', 'invoicedate', 'price', 'customer_id', 'country']
            
            col_mappings = {}
            cols_layout = st.columns(4)
            
            options = ["-- Select --"] + list(df.columns)
            
            for idx, col in enumerate(expected_cols):
                match = next((c for c in df.columns if str(c).lower().replace(' ', '') == col), None)
                if match is None and col == 'invoicedate':
                    match = next((c for c in df.columns if 'date' in str(c).lower()), None)
                if match is None and col == 'customer_id':
                    match = next((c for c in df.columns if 'id' in str(c).lower() or 'customer' in str(c).lower()), None)
                    
                default_idx = options.index(match) if match else 0
                
                with cols_layout[idx % 4]:
                    col_mappings[col] = st.selectbox(f"Map '{col}'", options, index=default_idx, key=f"map_{col}")
                
            if st.button("Process Data"):
                missing = [k for k, v in col_mappings.items() if v == "-- Select --"]
                if missing:
                    st.error(f"Missing column mappings for: {', '.join(missing)}")
                else:
                    with st.spinner("Validating and Processing Data..."):
                        try:
                            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            processed_data = process_uploaded_data(df, col_mappings, base_dir)
                            st.session_state.uploaded_processed_data = processed_data
                            st.success("Data successfully processed! Navigate to other tabs to view your insights.")
                        except Exception as e:
                            st.error(f"Validation/Processing Error: {e}")

    else:
        st.info("No file uploaded. Currently using demo dataset.")
        
    st.markdown("---")
    if st.button("Clear Uploaded Data and Return to Demo"):
        st.session_state.uploaded_processed_data = None
        st.session_state.last_uploaded = None
        if 'raw_df' in st.session_state:
            del st.session_state.raw_df
        st.success("Reverted to demo dataset.")

def show_overview(data):
    st.title("Business Overview")
    df = data['agg_trans']
    
    col1, col2, col3, col4 = st.columns(4)
    total_rev = df['revenue'].sum()
    total_orders = df['invoice'].nunique()
    total_items = df[df['quantity'] > 0]['quantity'].sum()
    returns = df[df['is_return']]['quantity'].sum()
    
    col1.metric("Total Net Revenue", f"${total_rev:,.2f}")
    col2.metric("Total Invoices", f"{total_orders:,}")
    col3.metric("Items Sold", f"{total_items:,.0f}")
    col4.metric("Items Returned", f"{abs(returns):,.0f}")
    
    st.subheader("Monthly Net Revenue")
    # Resample to monthly and sum revenue
    monthly_rev = df.set_index('invoicedate').resample('ME')['revenue'].sum().reset_index()
    fig = px.line(monthly_rev, x='invoicedate', y='revenue', title='Monthly Revenue Trend', color_discrete_sequence=[BURGUNDY])
    st.plotly_chart(fig, use_container_width=True)

def get_segment_name(cluster_id):
    mapping = {
        0: "Churned / One-Offs",
        1: "VIPs / Whales",
        2: "Loyal / Active"
    }
    return mapping.get(cluster_id, f"Cluster {cluster_id}")

def show_segmentation(data):
    st.title("Customer Segmentation")
    st.markdown("K-Means clustering applied to RFM (Recency, Frequency, Monetary) features.")
    
    profiles = data['profiles'].copy()
    profiles['Segment'] = profiles['cluster'].apply(get_segment_name)
    profiles.rename(columns={'aov': 'net_aov'}, inplace=True)
    # Reorder columns to show Segment first
    cols = ['cluster', 'Segment'] + [c for c in profiles.columns if c not in ['cluster', 'Segment']]
    
    st.markdown("*Note: 'net_aov' (Net Average Order Value) can be negative if a customer's returned monetary value exceeds their purchase value during the timeframe.*")
    st.dataframe(profiles[cols], use_container_width=True)
    
    clusters = data['clusters'].copy()
    clusters['Segment'] = clusters['cluster'].apply(get_segment_name)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(clusters, x='recency', y='monetary', color='Segment', 
                          title="Recency vs Net Monetary by Segment", 
                          color_discrete_sequence=[GOLD, BURGUNDY, "#2E4053"])
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        fig2 = px.scatter(clusters, x='frequency', y='monetary', color='Segment', 
                          title="Frequency vs Net Monetary by Segment",
                          color_discrete_sequence=[GOLD, BURGUNDY, "#2E4053"])
        st.plotly_chart(fig2, use_container_width=True)

def show_inactivity(data):
    st.title("Inactivity Prediction")
    st.markdown("Predicting future inactivity (0 purchases in outcome window) based on historical features.")
    
    if data.get('metrics') is None:
        st.info("You are viewing predictions for uploaded data. The model has generated risk scores for each customer, which you can view in the 'Customer Explorer' tab. Historical model evaluations (ROC-AUC, Confusion Matrix) are only available for the pre-trained demo dataset.")
        return
        
    metrics = data['metrics']
    best_model = max(metrics.keys(), key=lambda k: metrics[k]['roc_auc'])
    st.write(f"**Best Model Evaluated:** {best_model} (ROC-AUC: {metrics[best_model]['roc_auc']:.4f})")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Comparison (ROC-AUC)")
        aucs = {k: v['roc_auc'] for k, v in metrics.items()}
        fig = px.bar(x=list(aucs.keys()), y=list(aucs.values()), labels={'x':'Model', 'y':'ROC-AUC'}, color_discrete_sequence=[GOLD])
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader(f"Confusion Matrix ({best_model})")
        cm = metrics[best_model]['confusion_matrix']
        fig2 = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"),
                         x=['Active', 'Inactive'], y=['Active', 'Inactive'], color_continuous_scale="Reds")
        st.plotly_chart(fig2, use_container_width=True)
        
    st.info("SHAP Limitation: SHAP explanations are omitted. The pipeline dynamically selects the best model (e.g., LogisticRegression vs XGBoost). SHAP requires model-specific explainers (TreeExplainer vs LinearExplainer), and enforcing one globally adds deployment instability for edge-cases.")

def show_explorer(data):
    st.title("Customer Explorer")
    
    merged = data['rfm'].copy()
    if 'clusters' in data:
        merged = merged.merge(data['clusters'][['customer_id', 'cluster']], on='customer_id', how='left')
    if 'predictions' in data:
        preds = data['predictions'][['customer_id', 'inactivity_risk_score', 'is_inactive']].copy()
        preds['is_inactive'] = preds['is_inactive'].replace({-1: 'Unknown (Predicted Only)'})
        merged = merged.merge(preds, on='customer_id', how='left')
        
    st.markdown("Search and filter the customer base. Note: `is_inactive` represents confirmed future inactivity (for demo data) or 'Unknown' (for uploads). `inactivity_risk_score` is the model's predicted probability.")
    
    search_id = st.text_input("Search by Customer ID (Optional):")
    
    if search_id:
        try:
            filtered = merged[merged['customer_id'] == float(search_id)]
        except:
            filtered = merged[merged['customer_id'] == search_id]
    else:
        filtered = merged
        
    col1, col2 = st.columns(2)
    with col1:
        if 'cluster' in merged.columns:
            selected_cluster = st.multiselect("Filter by Cluster", options=sorted(merged['cluster'].dropna().unique()))
            if selected_cluster:
                filtered = filtered[filtered['cluster'].isin(selected_cluster)]
    with col2:
        if 'inactivity_risk_score' in merged.columns:
            risk_threshold = st.slider("Min Predicted Inactivity Risk Score", 0.0, 1.0, 0.0)
            filtered = filtered[filtered['inactivity_risk_score'] >= risk_threshold]
            
    st.dataframe(filtered, use_container_width=True)
    
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV Report", csv, "customer_report.csv", "text/csv")

def show_recommendations(data):
    st.title("Business Recommendations")
    st.markdown("Actionable strategies based on segment profiles. *Note: Recommendations are strategic guidelines and do not guarantee specific financial outcomes or precise churn reduction rates.*")
    
    st.subheader("Churned / One-Offs (Cluster 0)")
    st.write("- **Profile:** High Recency (last seen long ago), Low Frequency, Low Net Spend.")
    st.write("- **Action:** Deploy re-engagement campaigns (e.g., automated email discounts). If risk scores remain extremely high and past ROI is negative, exclude from expensive paid marketing channels.")
    
    st.subheader("Loyal / Active (Cluster 2)")
    st.write("- **Profile:** Low Recency, Moderate Frequency, Moderate/High Spend.")
    st.write("- **Action:** Nurture with loyalty programs, early access to new products, and personalized cross-selling. Maintain their engagement without sacrificing margins via unnecessary discounts.")
    
    st.subheader("VIPs / Whales (Cluster 1)")
    st.write("- **Profile:** Extremely High Frequency and Spend (Major distributors/B2B).")
    st.write("- **Action:** Assign dedicated account managers. Provide white-glove service, bulk order discounts, and volume pricing tiers. Protect this segment proactively.")

if __name__ == '__main__':
    main()
