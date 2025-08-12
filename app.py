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
        st.error("Please ensure the CSV file is in the root of your Hugging Face Space and has the correct name.")
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

# --- THIS IS THE ROBUSTNESS FIX ---
# Define the features we would ideally like to pre-select.
desired_default_features = ['MA_20', 'MA_50', 'RSI', 'Volume', 'Volatility_30D']
# Create the list of available options for the user to choose from.
available_options = [col for col in df.columns if col != target_col]
# Create the final default list by only including features that actually exist.
final_default_features = [feat for feat in desired_default_features if feat in available_options]

feature_cols = st.sidebar.multiselect(
    "Select Features (X)",
    options=available_options,
    default=final_default_features # Use the safe, filtered list
)

model_name = st.sidebar.selectbox(
    "Select ML Model",
    options=['Linear Regression', 'Random Forest', 'Gradient Boosting']
)

run_experiment_button = st.sidebar.button("Run Experiment", type="primary")


# --- 5. MAIN CONTENT TABS ---
tab1, tab2 = st.tabs(["📈 Exploratory Data Analysis (EDA)", "🤖 Model Training & Evaluation"])

with tab1:
    st.header("Exploratory Data Analysis")
    st.markdown("These charts provide a professional overview of the pre-processed and feature-engineered dataset.")

    st.subheader("Price and Volume Analysis (Last 2 Years)")
    df_subset = df.last('2Y').copy()
    df_subset['Volume_Color'] = np.where(df_subset['Adjusted Close'] >= df_subset['Open'], 'green', 'red')
    
    fig_price_vol = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig_price_vol.add_trace(go.Candlestick(x=df_subset.index, open=df_subset['Open'], high=df_subset['High'], low=df_subset['Low'], close=df_subset['Adjusted Close'], name='Price'), row=1, col=1)
    fig_price_vol.add_trace(go.Bar(x=df_subset.index, y=df_subset['Volume'], marker_color=df_subset['Volume_Color'], name='Volume'), row=2, col=1)
    fig_price_vol.update_layout(xaxis_rangeslider_visible=False, template="plotly_white")
    fig_price_vol.update_yaxes(title_text="Price (CAD)", row=1, col=1)
    fig_price_vol.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig_price_vol, use_container_width=True)

    st.subheader("Momentum Indicators (Last 2 Years)")
    fig_momentum = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig_momentum.add_trace(go.Scatter(x=df_subset.index, y=df_subset['MACD'], name='MACD', line=dict(color='blue')), row=1, col=1)
    fig_momentum.add_trace(go.Scatter(x=df_subset.index, y=df_subset['MACD_Signal'], name='Signal', line=dict(color='orange')), row=1, col=1)
    fig_momentum.add_trace(go.Bar(x=df_subset.index, y=df_subset['MACD_Hist'], name='Histogram', marker_color='rgba(150, 150, 150, 0.5)'), row=1, col=1)
    fig_momentum.add_trace(go.Scatter(x=df_subset.index, y=df_subset['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig_momentum.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", row=2, col=1)
    fig_momentum.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", row=2, col=1)
    fig_momentum.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_momentum.update_yaxes(title_text="MACD", row=1, col=1)
    fig_momentum.update_yaxes(title_text="RSI", row=2, col=1)
    st.plotly_chart(fig_momentum, use_container_width=True)
    
    st.subheader("Correlation Heatmap")
    corr_fig = px.imshow(df[['Adjusted Close', 'High', 'Low', 'Open', 'Volume', 'MA_20', 'MA_50', 'RSI']].corr(), text_auto=True, template="plotly_white")
    st.plotly_chart(corr_fig, use_container_width=True)

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

                st.subheader("Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("R-squared (R²)", f"{r2_score(y_test, y_pred):.3f}")
                col2.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, y_pred):.3f}")
                col3.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, y_pred):.3f}")

                st.subheader("Result Visualizations")
                perf_fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title='Model Performance: Actual vs. Predicted', trendline='ols', template="plotly_white")
                st.plotly_chart(perf_fig, use_container_width=True)

                residuals = y_test - y_pred
                resid_fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'}, title='Residuals Plot', template="plotly_white")
                resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(resid_fig, use_container_width=True)

                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
                    importance_fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Feature Importance', template="plotly_white")
                    st.plotly_chart(importance_fig, use_container_width=True)

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
