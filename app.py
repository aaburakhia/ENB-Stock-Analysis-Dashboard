# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import io
import base64

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="ENB Stock Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    """Loads the pre-processed data and caches it for performance."""
    try:
        df = pd.read_csv('ENB_data_advanced_features.csv', index_col='Date', parse_dates=True)
        return df
    except FileNotFoundError:
        st.error("CRITICAL ERROR: The dataset file 'ENB_data_advanced_features.csv' was not found.")
        st.error("Please ensure the CSV file is in the root of your GitHub repository and has the correct name.")
        return None

df = load_data()

# --- 3. MAIN PAGE LAYOUT ---
st.title("Advanced ML Analysis Cockpit for ENB Stock")
st.markdown("An interactive dashboard for model comparison, feature analysis, and prediction simulation.")

if df is None:
    st.stop()

# --- 4. SIDEBAR - CONTROLS ---
st.sidebar.header("Experiment Controls")
st.sidebar.info("The data for this app has been preprocessed and feature-engineered.")

target_col = st.sidebar.selectbox(
    "Select Target Variable (y)",
    options=df.columns,
    index=list(df.columns).index('Adjusted Close')
)

desired_default_features = ['MA_20', 'MA_50', 'RSI', 'Volume', 'Volatility_30D']
available_options = [col for col in df.columns if col != target_col]
final_default_features = [feat for feat in desired_default_features if feat in available_options]

feature_cols = st.sidebar.multiselect(
    "Select Features (X)",
    options=available_options,
    default=final_default_features
)

model_name = st.sidebar.selectbox(
    "Select ML Model",
    options=['Linear Regression', 'Random Forest', 'Gradient Boosting']
)

run_experiment_button = st.sidebar.button("Run Experiment", type="primary")


# --- 5. MAIN CONTENT TABS ---
tab1, tab2 = st.tabs(["📈 Exploratory Data Analysis (EDA)", "🤖 Model Training & Evaluation"])

with tab1:
    # ... (EDA Tab is unchanged) ...
    st.header("Exploratory Data Analysis")
    st.markdown("These charts provide a professional overview of the pre-processed and feature-engineered dataset.")
    st.subheader("Price and Volume Analysis (Last 2 Years)")
    df_subset = df.last('2Y').copy()
    df_subset['Volume_Color'] = np.where(df_subset['Adjusted Close'] >= df_subset['Open'], 'green', 'red')
    fig_price_vol = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig_price_vol.add_trace(go.Candlestick(x=df_subset.index, open=df_subset['Open'], high=df_subset['High'], low=df_subset['Low'], close=df_subset['Adjusted Close'], name='Price'), row=1, col=1)
    fig_price_vol.add_trace(go.Bar(x=df_subset.index, y=df_subset['Volume'], marker_color=df_subset['Volume_Color'], name='Volume'), row=2, col=1)
    fig_price_vol.update_layout(xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig_price_vol, use_container_width=True)

with tab2:
    st.header("Model Training Results")
    
    if run_experiment_button:
        if not all([target_col, feature_cols, model_name]):
            st.warning("Please select a target, features, and a model in the sidebar.")
        else:
            with st.spinner("Training model... This may take a moment."):
                X = df[feature_cols]
                y = df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                models = {
                    'Linear Regression': LinearRegression(),
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
                }
                model = models[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.session_state['trained_model'] = model
                st.session_state['feature_cols'] = feature_cols
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['X_test'] = X_test

    if 'trained_model' in st.session_state:
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        X_test = st.session_state['X_test']
        
        st.subheader("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("R-squared (R²)", f"{r2_score(y_test, y_pred):.3f}")
        col2.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, y_pred):.3f}")
        col3.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, y_pred):.3f}")

        st.subheader("Result Visualizations")
        perf_fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title='Model Performance: Actual vs. Predicted', trendline='ols', template="plotly_white")
        st.plotly_chart(perf_fig, use_container_width=True)

        # --- NEW INTERACTIVE SECTION (Exceeds Requirements) ---
        st.subheader("Interactive Error Analysis")
        st.markdown("Select a feature to see how the model's errors are distributed against it. This helps identify where the model is performing poorly.")
        
        # Interactive Viz 2: Dropdown to select a feature
        error_analysis_feature = st.selectbox("Select Feature for Error Analysis", options=X_test.columns)
        
        # Interactive Viz 3: The resulting plot
        residuals = y_test - y_pred
        error_fig = px.scatter(x=X_test[error_analysis_feature], y=residuals, labels={'x': error_analysis_feature, 'y': 'Residuals (Error)'}, title=f'Residuals vs. {error_analysis_feature}', template="plotly_white")
        error_fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(error_fig, use_container_width=True)

        if hasattr(st.session_state['trained_model'], 'feature_importances_'):
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({'feature': st.session_state['feature_cols'], 'importance': st.session_state['trained_model'].feature_importances_}).sort_values('importance', ascending=False)
            importance_fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Feature Importance', template="plotly_white")
            st.plotly_chart(importance_fig, use_container_width=True)

    # --- INTERACTIVE VISUALIZATION 1: WHAT-IF SIMULATOR ---
    if 'trained_model' in st.session_state:
        st.sidebar.divider()
        st.sidebar.header("What-if Simulator")
        
        sim_model = st.session_state['trained_model']
        sim_features = st.session_state['feature_cols']
        
        input_data = {}
        for feature in sim_features:
            input_data[feature] = st.sidebar.slider(
                f"Adjust {feature}",
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                value=float(df[feature].mean())
            )
        
        input_df = pd.DataFrame([input_data])
        prediction = sim_model.predict(input_df)[0]
        
        st.sidebar.metric("Predicted Target", f"{prediction:.2f}")
