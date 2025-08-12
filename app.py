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
from sklearn.preprocessing import StandardScaler
import joblib
import io
import base64
from datetime import datetime, timedelta

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="ENB Stock Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ENB_data_advanced_features.csv', index_col='Date', parse_dates=True)
        return df
    except FileNotFoundError:
        st.error("CRITICAL ERROR: The dataset file 'ENB_data_advanced_features.csv' was not found.")
        return None

@st.cache_data
def calculate_correlation_matrix(df, features):
    """Calculate correlation matrix for selected features"""
    return df[features].corr()

@st.cache_data
def train_all_models(X_train, y_train, X_test, y_test, rf_params, gb_params):
    """Train all models and return results"""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(**rf_params, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(**gb_params, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    return results

df = load_data()

# --- 3. MAIN PAGE LAYOUT ---
st.title("🚀 Advanced ML Analysis Cockpit for ENB Stock")
st.markdown("An interactive dashboard for model comparison, feature analysis, and prediction simulation with advanced analytics capabilities.")

if df is None:
    st.stop()

# --- 4. SIDEBAR - CONTROLS ---
st.sidebar.header("🎛️ Experiment Controls")
st.sidebar.info("The data for this app has been preprocessed and feature-engineered.")

target_col = st.sidebar.selectbox(
    "Select Target Variable (y)",
    options=df.columns,
    index=list(df.columns).index('Adjusted Close') if 'Adjusted Close' in df.columns else 0
)

desired_default_features = ['MA_20', 'MA_50', 'RSI', 'Volume', 'Volatility_30D']
available_options = [col for col in df.columns if col != target_col]
final_default_features = [feat for feat in desired_default_features if feat in available_options]

feature_cols = st.sidebar.multiselect(
    "Select Features (X)",
    options=available_options,
    default=final_default_features[:5] if len(final_default_features) >= 5 else final_default_features
)

# Enhanced model parameters
st.sidebar.subheader("🔧 Model Hyperparameters")

# Random Forest Parameters
st.sidebar.markdown("**Random Forest Settings:**")
rf_n_estimators = st.sidebar.slider("RF: Number of Trees", 50, 500, 100, 25)
rf_max_depth = st.sidebar.slider("RF: Max Depth", 3, 20, 10)

# Gradient Boosting Parameters  
st.sidebar.markdown("**Gradient Boosting Settings:**")
gb_n_estimators = st.sidebar.slider("GB: Number of Estimators", 50, 500, 100, 25)
gb_learning_rate = st.sidebar.slider("GB: Learning Rate", 0.01, 0.3, 0.1, 0.01)

rf_params = {'n_estimators': rf_n_estimators, 'max_depth': rf_max_depth}
gb_params = {'n_estimators': gb_n_estimators, 'learning_rate': gb_learning_rate}

run_experiment_button = st.sidebar.button("🚀 Run Complete Analysis", type="primary")

# --- 5. MAIN CONTENT TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Exploratory Data Analysis", "🤖 Model Training & Comparison", "🔍 Advanced Analytics"])

with tab1:
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Interactive exploration of the pre-processed and feature-engineered dataset.")

    # Interactive feature correlation heatmap
    st.subheader("🎯 Interactive Feature Correlation Analysis")
    
    if len(feature_cols) > 1:
        corr_features = st.multiselect(
            "Select features for correlation analysis:",
            options=feature_cols,
            default=feature_cols[:min(8, len(feature_cols))],
            key="corr_features"
        )
        
        if corr_features:
            corr_matrix = calculate_correlation_matrix(df, corr_features)
            
            # Interactive correlation heatmap
            fig_corr = px.imshow(
                corr_matrix, 
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig_corr.update_layout(
                title_font_size=16,
                template="plotly_white"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    st.subheader("📈 Price and Volume Analysis (Interactive)")
    
    # Time period selector
    time_periods = {
        "Last 6 Months": "6M",
        "Last Year": "1Y", 
        "Last 2 Years": "2Y",
        "All Data": "All"
    }
    
    selected_period = st.selectbox("Select time period:", list(time_periods.keys()), index=2)
    
    if time_periods[selected_period] == "All":
        df_subset = df.copy()
    else:
        df_subset = df.last(time_periods[selected_period]).copy()
    
    df_subset['Volume_Color'] = np.where(df_subset['Adjusted Close'] >= df_subset['Open'], 'green', 'red')
    
    fig_price_vol = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3],
        subplot_titles=('Price Action', 'Volume')
    )
    
    fig_price_vol.add_trace(
        go.Candlestick(
            x=df_subset.index, 
            open=df_subset['Open'], 
            high=df_subset['High'], 
            low=df_subset['Low'], 
            close=df_subset['Adjusted Close'], 
            name='Price'
        ), 
        row=1, col=1
    )
    
    fig_price_vol.add_trace(
        go.Bar(
            x=df_subset.index, 
            y=df_subset['Volume'], 
            marker_color=df_subset['Volume_Color'], 
            name='Volume'
        ), 
        row=2, col=1
    )
    
    fig_price_vol.update_layout(
        xaxis_rangeslider_visible=False, 
        template="plotly_white",
        title_text=f"ENB Stock Analysis - {selected_period}",
        height=600
    )
    
    st.plotly_chart(fig_price_vol, use_container_width=True)

with tab2:
    st.header("🤖 Model Training & Comparison")
    
    if run_experiment_button:
        if not all([target_col, feature_cols]):
            st.warning("⚠️ Please select a target and at least one feature in the sidebar.")
        else:
            with st.spinner("🔄 Training all models... This may take a moment."):
                # Prepare data
                X = df[feature_cols].dropna()
                y = df[target_col].dropna()
                
                # Align X and y indices
                common_index = X.index.intersection(y.index)
                X = X.loc[common_index]
                y = y.loc[common_index]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train all models
                results = train_all_models(X_train, y_train, X_test, y_test, rf_params, gb_params)
                
                # Store in session state
                st.session_state['results'] = results
                st.session_state['feature_cols'] = feature_cols
                st.session_state['y_test'] = y_test
                st.session_state['X_test'] = X_test
                st.session_state['target_col'] = target_col

    if 'results' in st.session_state:
        results = st.session_state['results']
        y_test = st.session_state['y_test']
        X_test = st.session_state['X_test']
        
        # Model comparison metrics
        st.subheader("📊 Model Performance Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, model_results in results.items():
            comparison_data.append({
                'Model': model_name,
                'R²': model_results['r2'],
                'MAE': model_results['mae'],
                'RMSE': model_results['rmse']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Interactive metric selector
        selected_metric = st.selectbox(
            "Select metric for comparison:",
            ['R²', 'MAE', 'RMSE'],
            help="R² (higher is better), MAE & RMSE (lower is better)"
        )
        
        # Create comparison chart
        fig_comparison = px.bar(
            comparison_df, 
            x='Model', 
            y=selected_metric,
            title=f'Model Comparison - {selected_metric}',
            color=selected_metric,
            color_continuous_scale='viridis'
        )
        fig_comparison.update_layout(template="plotly_white")
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Detailed metrics table
        st.dataframe(comparison_df.set_index('Model'), use_container_width=True)
        
        # Interactive model selector for detailed analysis
        st.subheader("🔍 Detailed Model Analysis")
        selected_model_name = st.selectbox(
            "Select model for detailed analysis:",
            list(results.keys())
        )
        
        selected_results = results[selected_model_name]
        y_pred = selected_results['predictions']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted scatter plot
            perf_fig = px.scatter(
                x=y_test, 
                y=y_pred,
                labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                title=f'{selected_model_name}: Actual vs. Predicted',
                trendline='ols',
                template="plotly_white"
            )
            # Add perfect prediction line
            min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            perf_fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], 
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                )
            )
            st.plotly_chart(perf_fig, use_container_width=True)
        
        with col2:
            # Residuals plot
            residuals = y_test - y_pred
            resid_fig = px.scatter(
                x=y_pred, 
                y=residuals,
                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                title=f'{selected_model_name}: Residuals Analysis',
                template="plotly_white"
            )
            resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(resid_fig, use_container_width=True)
        
        # Feature importance (for tree-based models)
        if hasattr(selected_results['model'], 'feature_importances_'):
            st.subheader("🎯 Feature Importance Analysis")
            importance_df = pd.DataFrame({
                'Feature': st.session_state['feature_cols'],
                'Importance': selected_results['model'].feature_importances_
            }).sort_values('Importance', ascending=True)
            
            importance_fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title=f'{selected_model_name}: Feature Importance',
                template="plotly_white"
            )
            st.plotly_chart(importance_fig, use_container_width=True)

    # What-if Simulator (Enhanced)
    if 'results' in st.session_state:
        st.subheader("🎮 Interactive What-If Simulator")
        
        sim_col1, sim_col2 = st.columns([2, 1])
        
        with sim_col1:
            st.markdown("**Adjust feature values to see predictions across all models:**")
            
            # Create input controls in columns
            input_cols = st.columns(min(3, len(st.session_state['feature_cols'])))
            input_data = {}
            
            for i, feature in enumerate(st.session_state['feature_cols']):
                with input_cols[i % len(input_cols)]:
                    input_data[feature] = st.slider(
                        f"{feature}",
                        min_value=float(df[feature].min()),
                        max_value=float(df[feature].max()),
                        value=float(df[feature].mean()),
                        key=f"sim_{feature}"
                    )
        
        with sim_col2:
            st.markdown("**Predictions:**")
            input_df = pd.DataFrame([input_data])
            
            sim_predictions = []
            for model_name, model_results in st.session_state['results'].items():
                prediction = model_results['model'].predict(input_df)[0]
                sim_predictions.append({
                    'Model': model_name,
                    'Prediction': prediction
                })
            
            sim_df = pd.DataFrame(sim_predictions)
            
            # Create prediction comparison chart
            sim_fig = px.bar(
                sim_df,
                x='Model',
                y='Prediction',
                title='Model Predictions Comparison',
                template="plotly_white"
            )
            st.plotly_chart(sim_fig, use_container_width=True)
            
            # Display values
            for _, row in sim_df.iterrows():
                st.metric(row['Model'], f"{row['Prediction']:.2f}")

with tab3:
    st.header("🔍 Advanced Analytics & Insights")
    
    if 'results' in st.session_state:
        # Prediction confidence analysis
        st.subheader("📈 Prediction Confidence Analysis")
        
        # Calculate prediction intervals for ensemble models
        selected_ensemble = st.selectbox(
            "Select ensemble model for confidence analysis:",
            ['Random Forest', 'Gradient Boosting']
        )
        
        if selected_ensemble in st.session_state['results']:
            model = st.session_state['results'][selected_ensemble]['model']
            
            # For tree-based models, we can get prediction variance
            if hasattr(model, 'estimators_'):
                # Get predictions from individual estimators
                individual_preds = np.array([
                    estimator.predict(X_test) for estimator in model.estimators_[:50]
                ])
                
                pred_mean = np.mean(individual_preds, axis=0)
                pred_std = np.std(individual_preds, axis=0)
                
                # Create confidence interval plot
                conf_df = pd.DataFrame({
                    'Actual': y_test.values,
                    'Predicted': pred_mean,
                    'Lower_CI': pred_mean - 1.96 * pred_std,
                    'Upper_CI': pred_mean + 1.96 * pred_std,
                    'Index': range(len(y_test))
                })
                
                fig_conf = go.Figure()
                
                # Add confidence interval
                fig_conf.add_trace(go.Scatter(
                    x=conf_df['Index'],
                    y=conf_df['Upper_CI'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False
                ))
                
                fig_conf.add_trace(go.Scatter(
                    x=conf_df['Index'],
                    y=conf_df['Lower_CI'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='95% Confidence Interval',
                    fillcolor='rgba(0,100,80,0.2)'
                ))
                
                # Add predictions and actual values
                fig_conf.add_trace(go.Scatter(
                    x=conf_df['Index'],
                    y=conf_df['Predicted'],
                    mode='lines',
                    name='Predicted',
                    line=dict(color='blue')
                ))
                
                fig_conf.add_trace(go.Scatter(
                    x=conf_df['Index'],
                    y=conf_df['Actual'],
                    mode='markers',
                    name='Actual',
                    marker=dict(color='red', size=4)
                ))
                
                fig_conf.update_layout(
                    title=f'{selected_ensemble}: Prediction Confidence Intervals',
                    xaxis_title='Test Sample Index',
                    yaxis_title=st.session_state['target_col'],
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_conf, use_container_width=True)
        
        # Error distribution analysis
        st.subheader("📊 Error Distribution Analysis")
        
        error_model = st.selectbox(
            "Select model for error analysis:",
            list(st.session_state['results'].keys()),
            key="error_model"
        )
        
        error_results = st.session_state['results'][error_model]
        errors = y_test - error_results['predictions']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error histogram
            fig_hist = px.histogram(
                x=errors,
                nbins=30,
                title=f'{error_model}: Error Distribution',
                labels={'x': 'Prediction Error', 'y': 'Frequency'},
                template='plotly_white'
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Q-Q plot for normality check
            from scipy import stats
            qq_data = stats.probplot(errors, dist="norm")
            
            fig_qq = px.scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                title=f'{error_model}: Q-Q Plot (Normality Check)',
                labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
                template='plotly_white'
            )
            
            # Add reference line
            fig_qq.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', dash='dash')
            ))
            
            st.plotly_chart(fig_qq, use_container_width=True)
        
        # Model interpretation insights
        st.subheader("🧠 Model Insights & Recommendations")
        
        # Calculate and display insights
        best_model = max(st.session_state['results'].items(), key=lambda x: x[1]['r2'])
        worst_model = min(st.session_state['results'].items(), key=lambda x: x[1]['r2'])
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.success(f"""
            **🏆 Best Performing Model: {best_model[0]}**
            - R² Score: {best_model[1]['r2']:.3f}
            - RMSE: {best_model[1]['rmse']:.3f}
            - This model explains {best_model[1]['r2']*100:.1f}% of the variance in {st.session_state['target_col']}
            """)
        
        with insight_col2:
            improvement_potential = (best_model[1]['r2'] - worst_model[1]['r2']) * 100
            st.info(f"""
            **📊 Performance Analysis:**
            - Performance gap: {improvement_potential:.1f}% R² difference
            - Models show {'high' if improvement_potential > 10 else 'moderate' if improvement_potential > 5 else 'low'} variance in performance
            - Consider ensemble methods for better results
            """)
        
    else:
        st.info("👆 Please run the experiment in the 'Model Training & Comparison' tab first to see advanced analytics.")
