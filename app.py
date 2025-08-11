# app.py - The Complete Streamlit ML Workflow Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="ML Workflow Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. DATA LOADING & CACHING ---
# We use @st.cache_data to load the data once and store it in memory.
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ENB_data_advanced_features.csv', index_col='Date', parse_dates=True)
        return df
    except FileNotFoundError:
        st.error("ERROR: Dataset not found. Please ensure 'ENB_data_advanced_features.csv' is in the same directory.")
        return None

df = load_data()

# --- 3. SIDEBAR - CONTROLS ---
st.sidebar.header("Experiment Controls")

if df is not None:
    # Requirement: Data Preprocessing (Explained)
    st.sidebar.info("The data for this app has been preprocessed and feature-engineered in a separate notebook.")

    target_col = st.sidebar.selectbox("Select Target Variable (y)", options=df.columns, index=list(df.columns).index('Adjusted Close'))
    
    feature_cols = st.sidebar.multiselect("Select Features (X)", options=[col for col in df.columns if col != target_col], default=['MA_20', 'MA_50', 'RSI', 'Volume', 'Volatility_30D'])
    
    # Requirement: 3 ML Models
    model_name = st.sidebar.selectbox("Select ML Model", options=['Linear Regression', 'Random Forest', 'Gradient Boosting'])

    run_experiment_button = st.sidebar.button("Run Experiment", type="primary")

else:
    st.sidebar.warning("Please upload the dataset to proceed.")

# --- 4. MAIN PAGE - LAYOUT & TABS ---
st.title("Advanced ML Analysis Cockpit for ENB Stock")
st.markdown("An interactive dashboard for model comparison, feature analysis, and prediction simulation.")

if df is not None:
    tab1, tab2 = st.tabs(["Exploratory Data Analysis (EDA)", "Model Training & Evaluation"])

    # --- EDA TAB ---
    with tab1:
        st.header("Exploratory Data Analysis")
        st.markdown("These charts provide a professional overview of the pre-processed and feature-engineered dataset.")
        
        # Chart 1 & 2: Professional Price and Volume Chart
        st.subheader("Price and Volume Analysis")
        df_subset = df.last('2Y').copy()
        df_subset['Volume_Color'] = np.where(df_subset['Adjusted Close'] >= df_subset['Open'], 'green', 'red')
        fig_price_vol = go.Figure()
        fig_price_vol.add_trace(go.Candlestick(x=df_subset.index, open=df_subset['Open'], high=df_subset['High'], low=df_subset['Low'], close=df_subset['Adjusted Close'], name='Price'))
        st.plotly_chart(fig_price_vol, use_container_width=True)

        # Chart 3 & 4: Momentum Indicators
        st.subheader("Momentum Indicators")
        fig_momentum = go.Figure()
        fig_momentum.add_trace(go.Scatter(x=df_subset.index, y=df_subset['MACD'], name='MACD'))
        fig_momentum.add_trace(go.Scatter(x=df_subset.index, y=df_subset['RSI'], name='RSI'))
        st.plotly_chart(fig_momentum, use_container_width=True)
        
        # Chart 5: Correlation Heatmap
        st.subheader("Correlation Heatmap")
        corr_fig = px.imshow(df[feature_cols + [target_col]].corr(), text_auto=True)
        st.plotly_chart(corr_fig, use_container_width=True)

    # --- MODEL EVALUATION TAB ---
    with tab2:
        st.header("Model Training Results")
        
        if run_experiment_button:
            if not all([target_col, feature_cols, model_name]):
                st.warning("Please select a target, features, and a model in the sidebar.")
            else:
                with st.spinner("Training model... This may take a moment."):
                    # 1. Prepare Data
                    X = df[feature_cols]
                    y = df[target_col]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # 2. Select and Train Model
                    models = {
                        'Linear Regression': LinearRegression(),
                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
                    }
                    model = models[model_name]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Store the trained model in session state for the simulator
                    st.session_state['trained_model'] = model
                    st.session_state['feature_cols'] = feature_cols

                    # 3. Display Results
                    st.subheader("Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R-squared (R²)", f"{r2_score(y_test, y_pred):.3f}")
                    col2.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, y_pred):.3f}")
                    col3.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, y_pred):.3f}")

                    st.subheader("Result Visualizations")
                    # Chart 6: Performance Plot
                    perf_fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title='Model Performance: Actual vs. Predicted', trendline='ols')
                    st.plotly_chart(perf_fig, use_container_width=True)

                    # Chart 7: Residuals Plot
                    residuals = y_test - y_pred
                    resid_fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'}, title='Residuals Plot')
                    resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(resid_fig, use_container_width=True)

                    # Chart 8: Feature Importance
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
                        importance_fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Feature Importance')
                        st.plotly_chart(importance_fig, use_container_width=True)

        # --- INTERACTIVE VISUALIZATION 1: WHAT-IF SIMULATOR ---
        if 'trained_model' in st.session_state:
            st.sidebar.divider()
            st.sidebar.header("What-if Simulator")
            
            # Retrieve the stored model and features
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
            
            # Create a DataFrame from the inputs and predict
            input_df = pd.DataFrame([input_data])
            prediction = sim_model.predict(input_df)[0]
            
            st.sidebar.success(f"Predicted Target: {prediction:.2f}")

else:
    st.error("Dataset could not be loaded. Please check the file path and format.")