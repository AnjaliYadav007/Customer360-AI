# Customer360 AI

Customer360 AI is an advanced, reusable customer intelligence platform for retail businesses. 
It analyzes historical transaction data to understand purchasing behavior, segments customers by value and engagement, and predicts future customer inactivity.

## Project Overview
This platform ingests raw transactional data and builds a fully automated pipeline encompassing data cleaning, feature engineering, predictive modeling, and a highly interactive analytics dashboard.

### Core Features
- **Data Preprocessing & Validation**: robust cleaning pipelines to handle exact duplicates, bad-debt negatives, mixed-line returns, and missing values.
- **RFM Customer Segmentation**: extracts Recency, Frequency, and Monetary value, scales features using Yeo-Johnson transformations, and applies K-Means clustering (k=3 optimal).
- **Inactivity Prediction**: a leakage-safe, chronologically evaluated classification model (Logistic Regression achieved ROC-AUC 0.8051) to predict 90-day future inactivity.
- **Interactive Dashboard**: A responsive Streamlit web application providing Business KPIs, Segment Profiles, Predictive Distributions, and a Customer Explorer with CSV export.

## Dataset Attribution
The initial modeling is built on the **Online Retail II Data Set**, sourced from the UCI Machine Learning Repository.
*Citation: Chen, Daqing. (2019). Online Retail II. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D.*

## Technology Stack
- **Data Processing**: Pandas (with `calamine` engine for rapid Excel parsing), NumPy
- **Machine Learning**: Scikit-Learn, XGBoost
- **Data Visualization**: Plotly Express
- **Web Application**: Streamlit

## Setup & Execution

### 1. Environment Setup
Create a virtual environment and install the required packages:
```bash
python -m venv .venv
# Activate on Windows:
.\.venv\Scripts\Activate.ps1
# Activate on Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install python-calamine # For fast excel reading
```

### 2. Run the Pipelines
To reproduce the processed data and models from the raw dataset (`data/raw/online_retail_II.xlsx`):
```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/customer_segmentation.py
python src/inactivity_prediction.py
python src/validate_data.py
```

### 3. Launch the Dashboard
Start the Streamlit application locally:
```bash
streamlit run app/streamlit_app.py
```

## Model Evaluation Results
- **Segmentation (K-Means)**: 3 optimal clusters separating Churned/One-Offs, Active/Loyal, and VIP/Whales (Silhouette Score: 0.4571).
- **Inactivity Prediction (Logistic Regression)**: Test set ROC-AUC of **0.8051**, validating that historical RFM attributes reliably predict future engagement.

## Limitations & Considerations
- **SHAP Integration**: Excluded due to architecture flexibility (dynamic model selection limits the ability to hardcode `TreeExplainer` vs `LinearExplainer`).
- **Temporal Prediction Limitation**: While evaluated chronologically, a deployed prediction system requires continuously advancing observation windows, rather than relying strictly on the single historical snapshot built here.
- **Business Recommendations**: Guidance provided in the app is strategic and does not guarantee strict financial outcomes.

## Deployment (Streamlit Community Cloud)
To deploy this app publicly:
1. Push this repository to GitHub (ensure `.gitignore` excludes the virtual environment, caches, and potentially the raw large data file if over 100MB; use `git-lfs` if necessary).
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/), link your GitHub account, and deploy `app/streamlit_app.py`.
3. Streamlit Cloud will automatically install dependencies listed in `requirements.txt`.
