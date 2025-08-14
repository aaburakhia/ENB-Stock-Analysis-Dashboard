# app.py 

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="ML Analyst Cockpit",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    """Loads the pre-processed, feature-engineered data and caches it."""
    try:
        # We now load the file produced by our definitive notebook
        df = pd.read_csv('ENB_data_binary_classification.csv', index_col='Date', parse_dates=True)
        return df
    except FileNotFoundError:
        st.error("CRITICAL ERROR: The dataset file 'ENB_data_binary_classification.csv' was not found.")
        st.error("Please run the `binary_classification_analysis.ipynb` notebook first to generate this file.")
        return None

df = load_data()

# --- 3. MAIN PAGE LAYOUT ---
st.title("📈 Stock Direction Classifier: An Analyst's Cockpit")
st.markdown("An interactive tool for feature selection and model comparison to predict next-day market direction.")

if df is None:
    st.stop()

# --- 4. SIDEBAR - INTERACTIVE CONTROLS ---
st.sidebar.header("🔬 Experiment Controls")
st.sidebar.info("Select a set of features and click 'Run Analysis' to train and evaluate four different classification models.")

# Interactive Viz 1: Feature Selector
# The feature columns are all columns except the 'Target'
feature_cols_options = [col for col in df.columns if col != 'Target']
feature_cols = st.sidebar.multiselect(
    "Select Features (X) to Use",
    options=feature_cols_options,
    default=feature_cols_options
)

run_analysis_button = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# --- 5. MAIN CONTENT ---
if run_analysis_button:
    if not feature_cols:
        st.warning("Please select at least one feature to run the analysis.")
    else:
        with st.spinner("Running experiments... This may take a moment."):
            # --- DATA PREPARATION ---
            # The data is already prepared, we just need to select the features and split it.
            df_final = df[feature_cols + ['Target']]

            split_index = int(len(df_final) * 0.8)
            df_train = df_final.iloc[:split_index]
            df_test = df_final.iloc[split_index:]

            X_train, y_train = df_train[feature_cols], df_train['Target']
            X_test, y_test = df_test[feature_cols], df_test['Target']

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

            # --- MODEL TRAINING ---
            models = {
                'Logistic Regression (Baseline)': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'Support Vector Machine (SVM)': SVC(random_state=42)
            }
            
            results = {}
            for name, model in models.items():
                model.fit(X_train_resampled, y_train_resampled)
                predictions = model.predict(X_test_scaled)
                report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
                cm = confusion_matrix(y_test, predictions, labels=[0, 1]) # 0=Else, 1=Up
                results[name] = {'report': report, 'cm': cm}

            st.session_state['results'] = results

# --- DISPLAY RESULTS ---
if 'results' in st.session_state:
    results = st.session_state['results']
    
    st.header("📊 Model Performance Comparison")
    st.markdown("This table shows the detailed performance of each model based on your selected features.")

    # Interactive Viz 2: Metric Selector
    metric_to_plot = st.selectbox("Select Metric to Visualize", options=['Precision', 'Recall', 'F1-Score'])
    class_to_plot_map = {'Up': 1, 'Else': 0}
    class_to_plot_name = st.selectbox("Select Class to Visualize", options=['Up', 'Else'])
    class_to_plot_label = class_to_plot_map[class_to_plot_name]

    # Interactive Viz 3: The Comparison Chart
    plot_data = []
    for name, result in results.items():
        # The report keys are strings '0' and '1', so we convert our label to a string
        report_class_key = str(class_to_plot_label)
        plot_data.append({
            'Model': name,
            'Metric': result['report'][report_class_key][metric_to_plot.lower().replace('-','')]
        })
    plot_df = pd.DataFrame(plot_data)
    
    fig = px.bar(plot_df, x='Model', y='Metric', color='Model',
                 title=f"{class_to_plot_name} - {metric_to_plot} Comparison", text_auto='.3f')
    fig.update_layout(yaxis_title=metric_to_plot)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Reports & Confusion Matrices")
    for name, result in results.items():
        st.markdown(f"#### {name}")
        
        report = result['report']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Accuracy", f"{report['accuracy']:.2%}")
        col2.metric("'Up' Precision", f"{report['1']['precision']:.2%}")
        col3.metric("'Up' Recall", f"{report['1']['recall']:.2%}")
        col4.metric("'Up' F1-Score", f"{report['1']['f1-score']:.2%}")
        
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(result['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=['Else', 'Up'], yticklabels=['Else', 'Up'], ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix')
        st.pyplot(fig)

    st.info("""
    **How to use this dashboard:**
    - A trader who wants to avoid false positives (i.e., only act on strong "Up" signals) should look for the model and feature set with the highest **Precision** for the 'Up' class.
    - A trader who wants to catch as many "Up" days as possible should look for the model and feature set with the highest **Recall** for the 'Up' class.
    """)
else:
    st.info("Select features in the sidebar and click 'Run Analysis' to train the models and see the results.")
