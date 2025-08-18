import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve, precision_score, recall_score
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
import optuna
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Stock Market Analysis & Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .model-results {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

@st.cache_data
def load_data():
    """Load both raw OHLCV data and preprocessed binary classification data"""
    try:
        # Load raw OHLCV data
        raw_df = pd.read_csv("ENB_TO_2000_2025_clean.csv")
        if 'Date' in raw_df.columns:
            raw_df['Date'] = pd.to_datetime(raw_df['Date'])
            raw_df.set_index('Date', inplace=True)
        elif raw_df.index.name != 'Date':
            raw_df.index = pd.to_datetime(raw_df.index)
        
        # Load preprocessed data for modeling
        processed_df = pd.read_csv("ENB_data_binary_classification.csv", index_col=0)
        processed_df.index = pd.to_datetime(processed_df.index)
        
        return raw_df, processed_df
    except FileNotFoundError as e:
        st.error(f"Dataset not found! Please ensure the CSV files are in the working directory. Error: {str(e)}")
        return None, None

# ============= NEW EDA FUNCTIONS =============

def create_price_overview_plot(df):
    """Create comprehensive price overview with OHLC and volume"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('ENB Stock Price (OHLC)', 'Trading Volume'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # OHLC Plot
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', 
                            line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['High'], name='High', 
                            line=dict(color='green', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Low'], name='Low', 
                            line=dict(color='red', width=1, dash='dot')), row=1, col=1)
    
    # Volume Plot
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                        marker_color='lightblue', opacity=0.7), row=2, col=1)
    
    fig.update_layout(
        title="ENB Stock Analysis: Price Movement & Trading Activity",
        height=600,
        template="plotly_white",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($CAD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_price_distribution_analysis(df):
    """Analyze price distributions and returns"""
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Distribution', 'Daily Returns Distribution', 
                       'Price vs Volume', 'Monthly Returns Box Plot'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "box"}]]
    )
    
    # Price Distribution
    fig.add_trace(go.Histogram(x=df['Close'], nbinsx=50, name='Close Price', 
                              marker_color='skyblue'), row=1, col=1)
    
    # Daily Returns Distribution
    fig.add_trace(go.Histogram(x=df['Daily_Return'].dropna(), nbinsx=50, 
                              name='Daily Returns (%)', marker_color='orange'), row=1, col=2)
    
    # Price vs Volume Scatter
    fig.add_trace(go.Scatter(x=df['Volume'], y=df['Close'], mode='markers',
                            name='Price vs Volume', marker=dict(size=4, opacity=0.6)), row=2, col=1)
    
    # Monthly Returns Box Plot
    df['Month'] = df.index.month
    for month in sorted(df['Month'].unique()):
        month_data = df[df['Month'] == month]['Daily_Return'].dropna()
        fig.add_trace(go.Box(y=month_data, name=f'Month {month}', 
                            showlegend=False), row=2, col=2)
    
    fig.update_layout(height=800, template="plotly_white", 
                     title="ENB Stock: Price and Return Distributions")
    
    return fig

def create_volatility_analysis(df):
    """Analyze volatility patterns and trends"""
    # Calculate rolling volatility (30-day)
    df['Volatility_30d'] = df['Close'].pct_change().rolling(window=30).std() * np.sqrt(252) * 100
    
    # Calculate price ranges
    df['Daily_Range'] = ((df['High'] - df['Low']) / df['Close']) * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('30-Day Rolling Volatility (%)', 'Daily Price Range (%)'),
        vertical_spacing=0.15
    )
    
    # Volatility Plot
    fig.add_trace(go.Scatter(x=df.index, y=df['Volatility_30d'], 
                            name='30-Day Volatility', line=dict(color='red', width=2)), row=1, col=1)
    
    # Add volatility threshold line
    mean_volatility = df['Volatility_30d'].mean()
    fig.add_hline(y=mean_volatility, line_dash="dash", line_color="gray", 
                  annotation_text=f"Avg: {mean_volatility:.1f}%", row=1, col=1)
    
    # Daily Range Plot
    fig.add_trace(go.Scatter(x=df.index, y=df['Daily_Range'], 
                            name='Daily Range', line=dict(color='purple', width=1)), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_white", 
                     title="ENB Stock: Volatility Analysis for Investment Timing")
    fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
    fig.update_yaxes(title_text="Daily Range (%)", row=2, col=1)
    
    return fig

def create_seasonal_analysis(df):
    """Analyze seasonal patterns in stock performance"""
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Quarter'] = df.index.quarter
    df['DayOfWeek'] = df.index.dayofweek
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Returns by Month', 'Average Returns by Quarter', 
                       'Average Returns by Day of Week', 'Yearly Performance'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Monthly Returns
    monthly_returns = df.groupby('Month')['Daily_Return'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    colors_monthly = ['red' if x < 0 else 'green' for x in monthly_returns.values]
    
    fig.add_trace(go.Bar(x=month_names, y=monthly_returns.values, 
                        name='Monthly Avg Return', marker_color=colors_monthly), row=1, col=1)
    
    # Quarterly Returns
    quarterly_returns = df.groupby('Quarter')['Daily_Return'].mean()
    colors_quarterly = ['red' if x < 0 else 'green' for x in quarterly_returns.values]
    
    fig.add_trace(go.Bar(x=[f'Q{i}' for i in quarterly_returns.index], 
                        y=quarterly_returns.values, name='Quarterly Avg Return',
                        marker_color=colors_quarterly), row=1, col=2)
    
    # Day of Week Returns
    dow_returns = df.groupby('DayOfWeek')['Daily_Return'].mean()
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    colors_dow = ['red' if x < 0 else 'green' for x in dow_returns.values]
    
    fig.add_trace(go.Bar(x=dow_names, y=dow_returns.values, 
                        name='Day of Week Avg Return', marker_color=colors_dow), row=2, col=1)
    
    # Yearly Performance
    yearly_returns = df.groupby('Year')['Close'].apply(lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) * 100)
    colors_yearly = ['red' if x < 0 else 'green' for x in yearly_returns.values]
    
    fig.add_trace(go.Bar(x=yearly_returns.index, y=yearly_returns.values, 
                        name='Yearly Return (%)', marker_color=colors_yearly), row=2, col=2)
    
    fig.update_layout(height=800, template="plotly_white", 
                     title="ENB Stock: Seasonal Investment Patterns")
    
    return fig

def create_technical_indicators_analysis(df):
    """Create technical analysis indicators"""
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)
    
    # RSI (simplified version)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price with Moving Averages', 'Bollinger Bands', 'RSI Indicator'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.4, 0.2]
    )
    
    # Price with Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', 
                            line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                            line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                            line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', 
                            line=dict(color='green', width=1)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', 
                            line=dict(color='blue', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                            line=dict(color='red', width=1, dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                            line=dict(color='red', width=1, dash='dash'), 
                            fill='tonexty', fillcolor='rgba(255,0,0,0.1)'), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', 
                            line=dict(color='purple', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="Overbought (70)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                  annotation_text="Oversold (30)", row=3, col=1)
    
    fig.update_layout(height=900, template="plotly_white", 
                     title="ENB Stock: Technical Analysis Indicators")
    
    return fig

def create_dividend_analysis(df):
    """Analyze dividend patterns and yield"""
    # This is a simplified version - you might need actual dividend data
    # For now, we'll estimate based on typical ENB dividend patterns
    
    # Create quarterly dividend estimates (ENB typically pays quarterly)
    quarterly_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='Q')
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Stock Price vs Estimated Dividend Yield', 'Investment Value Growth'),
        vertical_spacing=0.15
    )
    
    # Estimated dividend yield (inverse relationship with price)
    df['Est_Dividend_Yield'] = 50 / df['Close']  # Simplified estimation
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Stock Price ($)', 
                            yaxis='y1', line=dict(color='blue', width=2)), row=1, col=1)
    
    # Add secondary y-axis for dividend yield
    fig.add_trace(go.Scatter(x=df.index, y=df['Est_Dividend_Yield'], 
                            name='Est. Dividend Yield (%)', 
                            line=dict(color='green', width=2), yaxis='y2'), row=1, col=1)
    
    # Investment growth simulation
    initial_investment = 10000
    df['Investment_Value'] = initial_investment * (df['Close'] / df['Close'].iloc[0])
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Investment_Value'], 
                            name='$10,000 Investment Value', 
                            line=dict(color='gold', width=2)), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_white", 
                     title="ENB Stock: Dividend Analysis and Investment Growth")
    
    # Update y-axes
    fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Investment Value ($)", row=2, col=1)
    
    return fig

def create_risk_analysis(df):
    """Analyze risk metrics and drawdowns"""
    # Calculate returns
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    
    # Calculate drawdown
    rolling_max = df['Close'].expanding().max()
    df['Drawdown'] = (df['Close'] - rolling_max) / rolling_max * 100
    
    # Risk metrics
    annual_return = df['Daily_Return'].mean() * 252 * 100
    annual_volatility = df['Daily_Return'].std() * np.sqrt(252) * 100
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cumulative Returns', 'Drawdown Analysis', 
                       'Risk-Return Profile', 'Return Distribution'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # Cumulative Returns
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Return'] * 100, 
                            name='Cumulative Return (%)', 
                            line=dict(color='green', width=2)), row=1, col=1)
    
    # Drawdown
    fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown'], 
                            name='Drawdown (%)', 
                            line=dict(color='red', width=2), 
                            fill='tozeroy', fillcolor='rgba(255,0,0,0.3)'), row=1, col=2)
    
    # Risk-Return scatter (rolling windows)
    window = 252  # 1 year
    rolling_returns = df['Daily_Return'].rolling(window).mean() * 252 * 100
    rolling_volatility = df['Daily_Return'].rolling(window).std() * np.sqrt(252) * 100
    
    fig.add_trace(go.Scatter(x=rolling_volatility, y=rolling_returns, 
                            mode='markers', name='Risk-Return Profile',
                            marker=dict(size=6, opacity=0.6, color='blue')), row=2, col=1)
    
    # Return Distribution
    fig.add_trace(go.Histogram(x=df['Daily_Return'] * 100, nbinsx=50, 
                              name='Daily Returns (%)', marker_color='purple'), row=2, col=2)
    
    fig.update_layout(height=800, template="plotly_white", 
                     title=f"ENB Stock: Risk Analysis (Sharpe Ratio: {sharpe_ratio:.2f})")
    
    return fig, annual_return, annual_volatility, sharpe_ratio

def create_investment_insights(df):
    """Generate actionable investment insights"""
    # Calculate key metrics
    df['Daily_Return'] = df['Close'].pct_change()
    recent_price = df['Close'].iloc[-1]
    avg_price_1y = df['Close'].tail(252).mean()
    volatility_30d = df['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100
    
    # Best/worst months
    df['Month'] = df.index.month
    monthly_returns = df.groupby('Month')['Daily_Return'].mean() * 100
    best_month = monthly_returns.idxmax()
    worst_month = monthly_returns.idxmin()
    
    # Recent performance
    performance_1m = ((df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21]) * 100
    performance_3m = ((df['Close'].iloc[-1] - df['Close'].iloc[-63]) / df['Close'].iloc[-63]) * 100
    performance_1y = ((df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252]) * 100
    
    insights = {
        'current_price': recent_price,
        'vs_1y_avg': ((recent_price - avg_price_1y) / avg_price_1y) * 100,
        'volatility_30d': volatility_30d,
        'best_month': best_month,
        'worst_month': worst_month,
        'perf_1m': performance_1m,
        'perf_3m': performance_3m,
        'perf_1y': performance_1y,
        'avg_daily_volume': df['Volume'].tail(30).mean(),
        'price_trend': 'Upward' if df['Close'].tail(10).mean() > df['Close'].tail(20).mean() else 'Downward'
    }
    
    return insights

# ============= END OF NEW EDA FUNCTIONS =============

def analyze_class_balance(y):
    """Analyze class distribution and determine best resampling strategy"""
    class_counts = np.bincount(y)
    minority_count = min(class_counts)
    majority_count = max(class_counts)
    imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
    
    return {
        'minority_count': minority_count,
        'majority_count': majority_count,
        'imbalance_ratio': imbalance_ratio,
        'total_samples': len(y),
        'class_distribution': dict(zip(np.unique(y), class_counts))
    }

def robust_resampling(X, y, method='auto', random_state=42):
    """
    Robust resampling with multiple fallback strategies
    """
    balance_info = analyze_class_balance(y)
    
    # If classes are already balanced (ratio < 1.5), don't resample
    if balance_info['imbalance_ratio'] < 1.5:
        return X, y, "No resampling needed - classes already balanced"
    
    # If minority class has very few samples, use simple oversampling
    if balance_info['minority_count'] < 6:
        try:
            ros = RandomOverSampler(random_state=random_state)
            X_res, y_res = ros.fit_resample(X, y)
            return X_res, y_res, "Random Over Sampling (insufficient samples for SMOTE)"
        except Exception as e:
            return X, y, f"Random Over Sampling failed: {str(e)}"
    
    # Determine optimal k_neighbors for SMOTE
    optimal_k = min(5, balance_info['minority_count'] - 1)
    
    resampling_methods = []
    
    if method == 'auto' or method == 'smote':
        resampling_methods.extend([
            ('SMOTE', SMOTE(random_state=random_state, k_neighbors=optimal_k)),
            ('SMOTE_k3', SMOTE(random_state=random_state, k_neighbors=min(3, optimal_k))),
            ('SMOTE_k1', SMOTE(random_state=random_state, k_neighbors=1))
        ])
    
    if method == 'auto' or method == 'adasyn':
        if balance_info['minority_count'] >= 2:
            resampling_methods.append(
                ('ADASYN', ADASYN(random_state=random_state, n_neighbors=optimal_k))
            )
    
    # Always include RandomOverSampler as final fallback
    resampling_methods.append(('RandomOverSampler', RandomOverSampler(random_state=random_state)))
    
    # Try each method in order
    for method_name, resampler in resampling_methods:
        try:
            X_res, y_res = resampler.fit_resample(X, y)
            return X_res, y_res, f"Successfully applied {method_name}"
        except Exception as e:
            continue
    
    # If all methods fail, return original data
    return X, y, "All resampling methods failed - using original data"

def get_class_weights(y):
    """Calculate class weights for imbalanced datasets"""
    try:
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    except Exception:
        return None

def create_correlation_heatmap(df):
    """Create an interactive correlation heatmap"""
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r"
    )
    fig.update_layout(height=600, width=800)
    return fig

def train_model(X_train, X_test, y_train, y_test, model_name, model, resampling_method='auto', use_class_weights=True):
    """Train a single model with robust resampling and error handling"""
    try:
        # Analyze class balance
        balance_info = analyze_class_balance(y_train)
        
        # Apply resampling if needed
        if resampling_method != 'none' and balance_info['imbalance_ratio'] > 1.2:
            X_train_res, y_train_res, resampling_msg = robust_resampling(X_train, y_train, resampling_method)
        else:
            X_train_res, y_train_res = X_train, y_train
            resampling_msg = "No resampling applied"
        
        # Apply class weights if model supports it and resampling didn't work well
        model_copy = None
        if use_class_weights and hasattr(model, 'class_weight'):
            class_weights = get_class_weights(y_train_res)
            if class_weights and balance_info['imbalance_ratio'] > 1.5:
                # Create a copy of the model with class weights
                model_params = model.get_params()
                model_params['class_weight'] = class_weights
                model_copy = type(model)(**model_params)
            else:
                model_copy = model
        else:
            model_copy = model
        
        # Train model
        model_copy.fit(X_train_res, y_train_res)
        
        # Predictions
        y_pred = model_copy.predict(X_test)
        y_pred_proba = None
        if hasattr(model_copy, 'predict_proba'):
            try:
                y_pred_proba = model_copy.predict_proba(X_test)[:, 1]
            except Exception:
                pass
        
        # Calculate metrics with error handling
        try:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        except Exception as e:
            st.warning(f"Error calculating metrics for {model_name}: {str(e)}")
            return None
        
        return {
            'model': model_copy,
            'name': model_name,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc_score': auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'resampling_info': resampling_msg,
            'class_balance': balance_info
        }
        
    except Exception as e:
        st.error(f"Error training {model_name}: {str(e)}")
        return None

def run_automl_optimization(X_train, X_test, y_train, y_test, model_type, metric='f1', n_trials=50):
    """Run AutoML optimization with robust error handling"""
    
    def objective(trial):
        try:
            if model_type == "Random Forest":
                n_estimators = trial.suggest_int('n_estimators', 10, 200)
                max_depth = trial.suggest_int('max_depth', 3, 20)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                    class_weight='balanced'
                )
            elif model_type == "Gradient Boosting":
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                max_depth = trial.suggest_int('max_depth', 3, 10)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42
                )
            elif model_type == "SVM":
                C = trial.suggest_float('C', 0.1, 10)
                gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
                kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
                model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, 
                          random_state=42, class_weight='balanced')
            else:  # Logistic Regression
                C = trial.suggest_float('C', 0.01, 10)
                model = LogisticRegression(C=C, random_state=42, max_iter=1000, 
                                        class_weight='balanced')
            
            # Use robust resampling
            X_train_res, y_train_res, _ = robust_resampling(X_train, y_train, 'auto')
            
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            
            # Return the specified metric with error handling
            try:
                if metric == 'f1':
                    return f1_score(y_test, y_pred, zero_division=0)
                elif metric == 'precision':
                    return precision_score(y_test, y_pred, zero_division=0)
                elif metric == 'recall':
                    return recall_score(y_test, y_pred, zero_division=0)
                else:
                    return accuracy_score(y_test, y_pred)
            except Exception:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    try:
        # Create study with error handling
        study = optuna.create_study(direction='maximize')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Train best model
        best_params = study.best_params
        
        if model_type == "Random Forest":
            best_model = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced')
        elif model_type == "Gradient Boosting":
            best_model = GradientBoostingClassifier(**best_params, random_state=42)
        elif model_type == "SVM":
            best_model = SVC(**best_params, probability=True, random_state=42, class_weight='balanced')
        else:
            best_model = LogisticRegression(**best_params, random_state=42, max_iter=1000, class_weight='balanced')
        
        result = train_model(X_train, X_test, y_train, y_test, f"{model_type} (AutoML-{metric.upper()})", best_model, 'auto')
        if result:
            result['automl_params'] = best_params
            result['automl_best_score'] = study.best_value
        return result
        
    except Exception as e:
        st.error(f"AutoML optimization failed for {model_type}: {str(e)}")
        return None

def create_roc_curve(model_results):
    """Create ROC curve for multiple models"""
    fig = go.Figure()
    
    # Get test data for ROC curve calculation
    df = st.session_state.data
    feature_cols = [col for col in df.columns if col != 'Target']
    y_test = df.iloc[int(len(df) * 0.8):]['Target']
    
    for result in model_results.values():
        if result and result['probabilities'] is not None:
            try:
                fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                auc_score = result['auc_score']
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f"{result['name']} (AUC: {auc_score:.3f})",
                    line=dict(width=2)
                ))
            except Exception:
                continue
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title="ROC Curves Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        template="plotly_white"
    )
    
    return fig

def create_confusion_matrix_plot(cm, model_name):
    """Create confusion matrix visualization"""
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title=f"Confusion Matrix - {model_name}",
        labels=dict(x="Predicted", y="Actual"),
        x=['Down Movement', 'Up Movement'],
        y=['Down Movement', 'Up Movement'],
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=400)
    return fig

def create_feature_importance_plot(model_result, feature_names):
    """Create feature importance plot"""
    if hasattr(model_result['model'], 'feature_importances_'):
        importances = model_result['model'].feature_importances_
    elif hasattr(model_result['model'], 'coef_'):
        importances = np.abs(model_result['model'].coef_[0])
    else:
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f"Feature Importance - {model_result['name']}",
        template="plotly_white"
    )
    fig.update_layout(height=400)
    return fig

# Main app header
st.markdown('<h1 class="main-header">📈 ENB Stock Analysis & Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎛️ Control Panel")
    st.markdown("---")
    
    # Tool Description
    st.subheader("📈 About This Tool")
    st.markdown("""
    **ENB Stock Analysis Dashboard** provides comprehensive analysis and machine learning predictions for Enbridge Inc. (ENB) stock.
    
    🔄 **Process:** Explore data → Understand patterns → Train models → Make decisions
    """)
    
    st.markdown("---")
    
    # Use Cases
    st.subheader("💼 Investment Strategies")
    
    with st.expander("🛡️ Conservative Investor", expanded=True):
        st.markdown("""
        **Focus: Capital Preservation**
        
        • Look for low volatility periods
        • Focus on dividend yield patterns  
        • Avoid investing during high drawdown periods
        • Use technical indicators for entry/exit points
        """)
    
    with st.expander("🚀 Growth Investor"):
        st.markdown("""
        **Focus: Capital Appreciation**
        
        • Identify seasonal patterns for timing
        • Look for breakout patterns in technical analysis
        • Focus on periods of positive momentum
        • Use predictive models for trend following
        """)
    
    with st.expander("⚖️ Income Investor"):
        st.markdown("""
        **Focus: Dividend Income**
        
        • Monitor dividend yield vs stock price
        • Look for price drops with stable dividends
        • Understand seasonal dividend patterns
        • Balance yield with capital preservation
        """)
    
    st.markdown("---")
    st.markdown("*💡 Use the analysis plots to understand when to buy, hold, or sell ENB stock.*")

# Load data
if st.session_state.data is None or st.session_state.raw_data is None:
    with st.spinner("Loading ENB stock data..."):
        st.session_state.raw_data, st.session_state.data = load_data()

if st.session_state.data is None or st.session_state.raw_data is None:
    st.stop()

raw_df = st.session_state.raw_data
df = st.session_state.data

# Main tabs - UPDATED with new Stock Analysis tab
tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Analysis", "🔍 Data Explorer", "🤖 Model Lab", "📈 Performance Center"])

# NEW TAB 1: Stock Analysis (EDA Plots)
with tab1:
    st.header("📊 ENB Stock Market Analysis")
    st.markdown("*Understanding ENB stock patterns for smarter investment decisions*")
    
    # Investment insights overview
    insights = create_investment_insights(raw_df)
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Current Price", f"${insights['current_price']:.2f}")
    with col2:
        delta_color = "normal" if insights['vs_1y_avg'] >= 0 else "inverse"
        st.metric("vs 1Y Avg", f"{insights['vs_1y_avg']:+.1f}%")
    with col3:
        st.metric("30D Volatility", f"{insights['volatility_30d']:.1f}%")
    with col4:
        st.metric("1M Return", f"{insights['perf_1m']:+.1f}%")
    with col5:
        st.metric("1Y Return", f"{insights['perf_1y']:+.1f}%")
    
    # Investment insights
    with st.expander("💡 Key Investment Insights", expanded=True):
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="insight-box">
            <h4>📅 Best Time to Invest</h4>
            <p><strong>Best Month:</strong> {month_names[insights['best_month']]}</p>
            <p><strong>Worst Month:</strong> {month_names[insights['worst_month']]}</p>
            <p><strong>Current Trend:</strong> {insights['price_trend']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            volatility_status = "Low" if insights['volatility_30d'] < 15 else "High" if insights['volatility_30d'] > 25 else "Medium"
            price_status = "Attractive" if insights['vs_1y_avg'] < -5 else "Expensive" if insights['vs_1y_avg'] > 10 else "Fair"
            
            st.markdown(f"""
            <div class="insight-box">
            <h4>🎯 Investment Recommendation</h4>
            <p><strong>Price Level:</strong> {price_status}</p>
            <p><strong>Volatility:</strong> {volatility_status}</p>
            <p><strong>Avg Daily Volume:</strong> {insights['avg_daily_volume']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Plot 1: Price Overview
    st.subheader("1. 📈 ENB Stock Price Overview & Trading Volume")
    fig1 = create_price_overview_plot(raw_df)
    st.plotly_chart(fig1, use_container_width=True)
    
    with st.expander("💡 What this tells investors"):
        st.markdown("""
        - **Price Trends**: Look for long-term upward trends for growth potential
        - **Volume Spikes**: High volume often indicates important price movements
        - **Support/Resistance**: Identify price levels where stock tends to bounce
        - **Investment Timing**: Buy during dips if long-term trend is positive
        """)
    
    # Plot 2: Distribution Analysis  
    st.subheader("2. 📊 Price & Returns Distribution Analysis")
    fig2 = create_price_distribution_analysis(raw_df)
    st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("💡 What this tells investors"):
        st.markdown("""
        - **Price Distribution**: Shows the most common price ranges for ENB
        - **Return Distribution**: Normal bell curve suggests predictable returns
        - **Price vs Volume**: High volume at certain prices indicates strong support/resistance
        - **Monthly Patterns**: Some months consistently show better/worse performance
        """)
    
    # Plot 3: Volatility Analysis
    st.subheader("3. ⚡ Volatility Analysis - Risk Assessment")
    fig3 = create_volatility_analysis(raw_df)
    st.plotly_chart(fig3, use_container_width=True)
    
    with st.expander("💡 What this tells investors"):
        st.markdown("""
        - **Low Volatility Periods**: Better for conservative investors and new positions
        - **High Volatility Periods**: Potential opportunities but higher risk
        - **Daily Range**: Shows how much the stock moves daily - important for day traders
        - **Investment Strategy**: Buy during low volatility, be cautious during high volatility
        """)
    
    # Plot 4: Seasonal Analysis
    st.subheader("4. 🗓️ Seasonal Investment Patterns")
    fig4 = create_seasonal_analysis(raw_df)
    st.plotly_chart(fig4, use_container_width=True)
    
    with st.expander("💡 What this tells investors"):
        st.markdown(f"""
        - **Best Month**: {month_names[insights['best_month']]} historically shows strongest returns
        - **Worst Month**: {month_names[insights['worst_month']]} typically shows weakest performance  
        - **Quarterly Patterns**: Some quarters consistently outperform others
        - **Day of Week**: Some days of the week show better average returns
        - **Strategy**: Time your investments based on historical seasonal patterns
        """)
    
    # Plot 5: Technical Analysis
    st.subheader("5. 📊 Technical Analysis - Buy/Sell Signals")
    fig5 = create_technical_indicators_analysis(raw_df)
    st.plotly_chart(fig5, use_container_width=True)
    
    with st.expander("💡 What this tells investors"):
        st.markdown("""
        - **Moving Averages**: When price is above all MAs, it's bullish; below all MAs is bearish
        - **Bollinger Bands**: Price near upper band may be overbought; near lower band may be oversold
        - **RSI Indicator**: Above 70 = potentially overbought (sell signal); Below 30 = potentially oversold (buy signal)
        - **Strategy**: Use these indicators to time your entry and exit points
        """)
    
    # Plot 6: Dividend Analysis
    st.subheader("6. 💰 Dividend Analysis & Investment Growth")
    fig6 = create_dividend_analysis(raw_df)
    st.plotly_chart(fig6, use_container_width=True)
    
    with st.expander("💡 What this tells investors"):
        st.markdown("""
        - **Dividend Yield**: Higher yield when stock price is lower (better value)
        - **Investment Growth**: Shows how $10,000 invested would have grown
        - **Income Strategy**: ENB is known for reliable quarterly dividends
        - **Timing**: Buy when dividend yield is high (stock price relatively low)
        """)
    
    # Plot 7: Risk Analysis  
    st.subheader("7. ⚠️ Risk Analysis - Maximum Drawdowns")
    fig7, annual_return, annual_vol, sharpe = create_risk_analysis(raw_df)
    st.plotly_chart(fig7, use_container_width=True)
    
    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annual Return", f"{annual_return:.1f}%")
    with col2:
        st.metric("Annual Volatility", f"{annual_vol:.1f}%")  
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col4:
        risk_level = "Low" if annual_vol < 20 else "High" if annual_vol > 30 else "Medium"
        st.metric("Risk Level", risk_level)
    
    with st.expander("💡 What this tells investors"):
        st.markdown(f"""
        - **Drawdown Analysis**: Shows the maximum loss from peak to trough  
        - **Risk-Return Profile**: ENB shows {risk_level.lower()} risk with {annual_return:.1f}% annual returns
        - **Sharpe Ratio**: {sharpe:.2f} indicates risk-adjusted performance ({'Good' if sharpe > 1 else 'Fair' if sharpe > 0.5 else 'Poor'})
        - **Strategy**: Understand maximum potential losses before investing
        """)
    
    # Investment Decision Framework
    st.subheader("8. 🎯 Investment Decision Framework")
    
    # Create decision matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 BUY Signals")
        buy_signals = []
        if insights['vs_1y_avg'] < -5:
            buy_signals.append("✅ Price below 1-year average (value opportunity)")
        if insights['volatility_30d'] < 20:
            buy_signals.append("✅ Low volatility period (lower risk)")
        if insights['perf_1m'] < -3:
            buy_signals.append("✅ Recent decline (potential rebound)")
        
        # Technical signals (simplified)
        current_rsi = 45  # Placeholder - you'd calculate this from actual RSI
        if current_rsi < 40:
            buy_signals.append("✅ RSI indicates oversold condition")
        
        if buy_signals:
            for signal in buy_signals:
                st.markdown(signal)
        else:
            st.markdown("⚠️ No strong buy signals currently")
    
    with col2:
        st.markdown("### 🔴 SELL/WAIT Signals")
        sell_signals = []
        if insights['vs_1y_avg'] > 15:
            sell_signals.append("⚠️ Price well above 1-year average (overvalued)")
        if insights['volatility_30d'] > 30:
            sell_signals.append("⚠️ High volatility period (higher risk)")
        if insights['perf_1m'] > 10:
            sell_signals.append("⚠️ Large recent gains (potential pullback)")
        
        if current_rsi > 70:
            sell_signals.append("⚠️ RSI indicates overbought condition")
        
        if sell_signals:
            for signal in sell_signals:
                st.markdown(signal)
        else:
            st.markdown("✅ No major sell signals currently")
    
    # Final recommendation
    st.markdown("---")
    buy_score = len(buy_signals) if 'buy_signals' in locals() else 0
    sell_score = len(sell_signals) if 'sell_signals' in locals() else 0
    
    if buy_score > sell_score:
        recommendation = "🟢 **CONSIDER BUYING** - Multiple positive signals present"
        rec_color = "success"
    elif sell_score > buy_score:
        recommendation = "🔴 **HOLD/WAIT** - Some caution signals present"
        rec_color = "warning"
    else:
        recommendation = "🟡 **NEUTRAL** - Mixed signals, wait for clearer trend"
        rec_color = "info"
    
    st.markdown(f"""
    <div class="alert alert-{rec_color}" role="alert">
    <h4>🎯 Overall Investment Recommendation</h4>
    <p>{recommendation}</p>
    <p><small>Remember: This is based on historical data analysis. Always do your own research and consider your risk tolerance.</small></p>
    </div>
    """, unsafe_allow_html=True)

# TAB 2: Original Data Explorer (simplified)
with tab2:
    st.header("🔍 Technical Data Explorer")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Dataset Overview")
        st.info(f"**Raw Data Shape:** {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
        st.info(f"**Processed Data Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.info(f"**Date Range:** {raw_df.index.min().date()} to {raw_df.index.max().date()}")
        
        # Class distribution analysis
        target_dist = df['Target'].value_counts()
        balance_info = analyze_class_balance(df['Target'].values)
        
        st.metric("Up Movement Cases", target_dist.get(1, 0))
        st.metric("Down Movement Cases", target_dist.get(0, 0))
        st.metric("Class Balance Ratio", f"{balance_info['imbalance_ratio']:.2f}")
        
        # Display balance status
        if balance_info['imbalance_ratio'] < 1.5:
            st.success("✅ Classes are well balanced")
        elif balance_info['imbalance_ratio'] < 3:
            st.warning("⚠️ Moderate class imbalance")
        else:
            st.error("❌ Severe class imbalance - resampling recommended")
    
    with col1:
        # Feature selection for visualization
        numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'Target']
        selected_feature = st.selectbox("Select Feature to Analyze", numeric_cols)
        
        # Time series plot
        if selected_feature:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[selected_feature],
                mode='lines',
                name=selected_feature,
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=f"Time Series: {selected_feature}",
                xaxis_title="Date",
                yaxis_title=selected_feature,
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Feature Correlation Analysis")
    fig_corr = create_correlation_heatmap(df)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Show raw data preview
    st.subheader("Raw OHLCV Data Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)
    
    st.subheader("Processed Features Data Preview")  
    st.dataframe(df.head(10), use_container_width=True)

# TAB 3: Model Lab (unchanged from original)
with tab3:
    st.header("🤖 Model Lab")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Model Configuration")
        
        # Feature selection
        feature_cols = [col for col in df.columns if col != 'Target']
        selected_features = st.multiselect(
            "Select Features",
            feature_cols,
            default=feature_cols[:5] if len(feature_cols) > 5 else feature_cols
        )
        
        # Resampling strategy
        resampling_options = {
            "Auto (Robust)": "auto",
            "SMOTE Only": "smote", 
            "ADASYN Only": "adasyn",
            "No Resampling": "none"
        }
        
        resampling_method = st.selectbox(
            "Class Balancing Strategy",
            list(resampling_options.keys()),
            index=0,
            help="Auto mode tries multiple methods and falls back gracefully"
        )
        
        # Model selection
        available_models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Naive Bayes": GaussianNB(),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
        }
        
        selected_models = st.multiselect(
            "Select Models",
            list(available_models.keys()),
            default=["Logistic Regression", "Random Forest"],
            help="Start with 1-2 models for faster training"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            use_class_weights = st.checkbox("Use Class Weights", value=True)
            
            # AutoML option
            use_automl = st.checkbox("🚀 Use AutoML Optimization")
            if use_automl:
                n_trials = st.slider("AutoML Trials", 10, 100, 30)
                automl_model = st.selectbox(
                    "Model for AutoML",
                    ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression"]
                )
                automl_metric = st.selectbox(
                    "Optimization Metric",
                    ["f1", "precision", "recall", "accuracy"]
                )
        
        # Train models button
        if st.button("🏋️ Train Models", type="primary"):
            if selected_features and selected_models:
                st.session_state.models_trained = True
                st.session_state.model_results = {}
    
    with col1:
        st.subheader("Training Results")
        
        if st.session_state.models_trained and selected_features:
            # Prepare data
            X = df[selected_features]
            y = df['Target']
            
            # Check for data issues
            if X.isnull().any().any():
                st.warning("⚠️ Found missing values in features. Please clean your data.")
                X = X.fillna(X.mean())
            
            # Train-test split (chronological)
            split_index = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            # Display data split info
            st.info(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
            
            # Analyze class balance
            train_balance = analyze_class_balance(y_train.values)
            if train_balance['imbalance_ratio'] > 3:
                st.warning(f"⚠️ High class imbalance detected (ratio: {train_balance['imbalance_ratio']:.2f})")
            
            # Scale features
            try:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            except Exception as e:
                st.error(f"Error scaling features: {str(e)}")
                st.stop()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            total_models = len(selected_models) + (1 if use_automl else 0)
            current_model = 0
            
            # Train regular models
            for model_name in selected_models:
                current_model += 1
                status_text.text(f"Training {model_name} ({current_model}/{total_models})...")
                
                model = available_models[model_name]
                result = train_model(
                    X_train_scaled, X_test_scaled, y_train, y_test, 
                    model_name, model, 
                    resampling_options[resampling_method], 
                    use_class_weights
                )
                
                if result:
                    st.session_state.model_results[model_name] = result
                    with results_container:
                        st.success(f"✅ {model_name} trained successfully")
                        st.text(f"   • {result['resampling_info']}")
                        st.text(f"   • Accuracy: {result['accuracy']:.3f}, F1: {result['f1_score']:.3f}")
                else:
                    with results_container:
                        st.error(f"❌ {model_name} training failed")
                
                progress_bar.progress(current_model / total_models)
            
            # Train AutoML model if selected
            if use_automl:
                current_model += 1
                status_text.text(f"Running AutoML for {automl_model} (optimizing {automl_metric.upper()})...")
                
                automl_result = run_automl_optimization(
                    X_train_scaled, X_test_scaled, y_train, y_test, 
                    automl_model, automl_metric, n_trials
                )
                
                if automl_result:
                    automl_key = f"{automl_model} (AutoML-{automl_metric.upper()})"
                    st.session_state.model_results[automl_key] = automl_result
                    with results_container:
                        st.success(f"✅ AutoML {automl_model} completed")
                        st.text(f"   • Best {automl_metric}: {automl_result['automl_best_score']:.3f}")
                        st.text(f"   • Final F1: {automl_result['f1_score']:.3f}")
                else:
                    with results_container:
                        st.error(f"❌ AutoML {automl_model} failed")
                
                progress_bar.progress(1.0)
            
            status_text.text("✅ Training completed!")
            
            # Display summary results
            if st.session_state.model_results:
                st.subheader("📊 Training Summary")
                
                results_data = []
                for result in st.session_state.model_results.values():
                    results_data.append({
                        'Model': result['name'],
                        'Accuracy': f"{result['accuracy']:.3f}",
                        'F1-Score': f"{result['f1_score']:.3f}",
                        'Precision': f"{result['precision']:.3f}",
                        'Recall': f"{result['recall']:.3f}",
                        'AUC': f"{result['auc_score']:.3f}" if result['auc_score'] else "N/A"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Show best performing model
                if results_data:
                    best_f1_model = max(st.session_state.model_results.values(), key=lambda x: x['f1_score'])
                    st.success(f"🏆 Best F1-Score: {best_f1_model['name']} ({best_f1_model['f1_score']:.3f})")
            else:
                st.warning("⚠️ No models were successfully trained.")
        
        elif not st.session_state.models_trained:
            st.info("👈 Configure your models and click 'Train Models' to see results here.")
            
            # Show data preview
            if not df.empty:
                st.subheader("Processed Data Preview")
                st.dataframe(df.head(), use_container_width=True)

# TAB 4: Performance Center (unchanged from original)
with tab4:
    st.header("📈 Performance Center")
    
    if not st.session_state.model_results:
        st.warning("⚠️ No models trained yet. Please go to the Model Lab tab to train some models first.")
    else:
        # Model comparison metrics
        st.subheader("📊 Model Comparison Dashboard")
        
        valid_results = {k: v for k, v in st.session_state.model_results.items() if v is not None}
        
        if not valid_results:
            st.error("❌ No valid model results found. Please retrain your models.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            # Best model metrics
            try:
                best_f1 = max(valid_results.values(), key=lambda x: x['f1_score'])
                best_acc = max(valid_results.values(), key=lambda x: x['accuracy'])
                best_precision = max(valid_results.values(), key=lambda x: x['precision'])
                best_recall = max(valid_results.values(), key=lambda x: x['recall'])
                
                with col1:
                    st.metric("🎯 Best F1-Score", f"{best_f1['f1_score']:.3f}", 
                             help=f"Model: {best_f1['name']}")
                with col2:
                    st.metric("🎯 Best Precision", f"{best_precision['precision']:.3f}",
                             help=f"Model: {best_precision['name']}")
                with col3:
                    st.metric("🎯 Best Recall", f"{best_recall['recall']:.3f}",
                             help=f"Model: {best_recall['name']}")
                with col4:
                    st.metric("📊 Models Trained", len(valid_results))
            except Exception as e:
                st.error(f"Error calculating best metrics: {str(e)}")
            
            # Performance comparison charts
            st.subheader("📈 Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC Curves
                try:
                    models_with_proba = {k: v for k, v in valid_results.items() if v['probabilities'] is not None}
                    if models_with_proba:
                        fig_roc = create_roc_curve(models_with_proba)
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.info("📊 ROC curves require models with probability predictions.")
                except Exception as e:
                    st.error(f"Error creating ROC curves: {str(e)}")
            
            with col2:
                # Performance metrics comparison
                try:
                    metrics_data = []
                    for result in valid_results.values():
                        metrics_data.append({
                            'Model': result['name'][:20] + "..." if len(result['name']) > 20 else result['name'],
                            'Accuracy': result['accuracy'],
                            'F1-Score': result['f1_score'],
                            'Precision': result['precision'],
                            'Recall': result['recall']
                        })
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        fig_metrics = px.bar(
                            metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                            x='Model', y='Score', color='Metric',
                            title="Performance Metrics Comparison",
                            barmode='group'
                        )
                        fig_metrics.update_xaxes(tickangle=45)
                        fig_metrics.update_layout(height=400)
                        st.plotly_chart(fig_metrics, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating metrics comparison: {str(e)}")
            
            # Individual model analysis
            st.subheader("🔍 Individual Model Analysis")
            
            model_names = list(valid_results.keys())
            selected_model_name = st.selectbox(
                "Select Model for Detailed Analysis",
                model_names,
                key="model_analysis_selector"
            )
            
            if selected_model_name and selected_model_name in valid_results:
                selected_result = valid_results[selected_model_name]
                
                # Model details
                with st.expander(f"📋 {selected_result['name']} - Model Details", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{selected_result['accuracy']:.3f}")
                        st.metric("Precision", f"{selected_result['precision']:.3f}")
                    
                    with col2:
                        st.metric("Recall", f"{selected_result['recall']:.3f}")
                        st.metric("F1-Score", f"{selected_result['f1_score']:.3f}")
                    
                    with col3:
                        if selected_result['auc_score']:
                            st.metric("AUC Score", f"{selected_result['auc_score']:.3f}")
                
                # Confusion Matrix
                col1, col2 = st.columns(2)
                
                with col1:
                    cm_fig = create_confusion_matrix_plot(
                        selected_result['confusion_matrix'], 
                        selected_result['name']
                    )
                    st.plotly_chart(cm_fig, use_container_width=True)
                
                with col2:
                    # Feature importance if available
                    if hasattr(selected_result['model'], 'feature_importances_') or hasattr(selected_result['model'], 'coef_'):
                        if 'selected_features' in locals():
                            fi_fig = create_feature_importance_plot(selected_result, selected_features)
                            if fi_fig:
                                st.plotly_chart(fi_fig, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model type.")
                
                # Classification Report
                with st.expander("📊 Detailed Classification Report"):
                    try:
                        if 'classification_report' in selected_result:
                            report_df = pd.DataFrame(selected_result['classification_report']).transpose()
                            
                            # Format numeric columns
                            for col in ['precision', 'recall', 'f1-score']:
                                if col in report_df.columns:
                                    report_df[col] = report_df[col].apply(
                                        lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else str(x)
                                    )
                            st.dataframe(report_df, use_container_width=True)
                        else:
                            st.warning("Classification report not available")
                    except Exception as e:
                        st.error(f"Error displaying classification report: {str(e)}")
                
                # AutoML specific information
                if 'automl_params' in selected_result:
                    with st.expander("🚀 AutoML Optimization Results"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Best Parameters:")
                            for param, value in selected_result['automl_params'].items():
                                st.text(f"• {param}: {value}")
                        
                        with col2:
                            st.subheader("Optimization Details:")
                            st.metric("Best Score", f"{selected_result['automl_best_score']:.3f}")
                            if 'n_trials' in locals():
                                st.metric("Trials Run", f"{n_trials}")
                            if 'automl_metric' in locals():
                                st.text(f"• Optimized for: {automl_metric.upper()}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <h4>🚀 ENB Stock Analysis & Prediction Dashboard</h4>
        <p><strong>Features:</strong> Comprehensive EDA • Investment Insights • Technical Analysis • ML Predictions</p>
        <p><strong>Built with:</strong> Streamlit • Scikit-learn • Plotly • Optuna</p>
        <p><em>Made by Ahmed Awad</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)
