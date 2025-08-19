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
    """Load both datasets"""
    try:
        # Load processed data for modeling
        df_processed = pd.read_csv("ENB_data_binary_classification.csv", index_col=0)
        df_processed.index = pd.to_datetime(df_processed.index)
        
        # Load raw OHLCV data for analysis
        df_raw = pd.read_csv("ENB_TO_2000_2025_clean.csv")
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_raw.set_index('Date', inplace=True)
        
        return df_processed, df_raw
    except FileNotFoundError as e:
        st.error(f"Dataset not found! Error: {str(e)}")
        return None, None

def analyze_class_balance(y):
    """Analyze class distribution"""
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
    """Robust resampling with fallback strategies"""
    balance_info = analyze_class_balance(y)
    
    if balance_info['imbalance_ratio'] < 1.5:
        return X, y, "No resampling needed - classes already balanced"
    
    if balance_info['minority_count'] < 6:
        try:
            ros = RandomOverSampler(random_state=random_state)
            X_res, y_res = ros.fit_resample(X, y)
            return X_res, y_res, "Random Over Sampling (insufficient samples for SMOTE)"
        except Exception as e:
            return X, y, f"Random Over Sampling failed: {str(e)}"
    
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
    
    resampling_methods.append(('RandomOverSampler', RandomOverSampler(random_state=random_state)))
    
    for method_name, resampler in resampling_methods:
        try:
            X_res, y_res = resampler.fit_resample(X, y)
            return X_res, y_res, f"Successfully applied {method_name}"
        except Exception as e:
            continue
    
    return X, y, "All resampling methods failed - using original data"

def get_class_weights(y):
    """Calculate class weights for imbalanced datasets"""
    try:
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    except Exception:
        return None

# Analysis plot functions
def create_price_trend_analysis(df_raw):
    """1. Overall price trend with volume"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.1,
                       subplot_titles=('Stock Price Trends', 'Trading Volume'),
                       row_heights=[0.7, 0.3])
    
    # Price trends
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['High'], name='High', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Low'], name='Low', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Adjusted Close'], name='Adj Close', line=dict(color='blue', width=2)), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df_raw.index, y=df_raw['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
    
    fig.update_layout(title="📈 Long-term Stock Performance & Trading Activity", height=600)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_return_distribution(df_raw):
    """2. Daily return distribution"""
    df_raw = df_raw.copy()
    df_raw['Daily_Return'] = df_raw['Adjusted Close'].pct_change() * 100
    
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Daily Returns Distribution', 'Daily Returns Over Time'))
    
    # Histogram
    fig.add_trace(go.Histogram(x=df_raw['Daily_Return'].dropna(), 
                              nbinsx=50, name='Returns Distribution', 
                              marker_color='lightblue'), row=1, col=1)
    
    # Time series
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Daily_Return'], 
                            mode='lines', name='Daily Returns',
                            line=dict(color='orange')), row=1, col=2)
    
    fig.update_layout(title="💰 Daily Returns Analysis - Risk Assessment", height=500)
    fig.update_xaxes(title_text="Return (%)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    
    return fig

def create_volatility_analysis(df_raw):
    """3. Volatility patterns"""
    df_raw['Daily_Return'] = df_raw['Adjusted Close'].pct_change()
    df_raw['Volatility_30D'] = df_raw['Daily_Return'].rolling(window=30).std() * np.sqrt(252) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Volatility_30D'],
                            mode='lines', name='30-Day Volatility',
                            line=dict(color='red', width=2)))
    
    # Add volatility zones
    fig.add_hline(y=df_raw['Volatility_30D'].quantile(0.75), line_dash="dash", 
                  annotation_text="High Volatility Zone", line_color="red")
    fig.add_hline(y=df_raw['Volatility_30D'].quantile(0.25), line_dash="dash", 
                  annotation_text="Low Volatility Zone", line_color="green")
    
    fig.update_layout(title="📊 Market Volatility - When to Be Cautious", 
                     xaxis_title="Date", yaxis_title="Volatility (%)", height=500)
    
    return fig

def create_seasonal_patterns(df_raw):
    """4. Seasonal investment patterns"""
    df_raw['Month'] = df_raw.index.month
    df_raw['Quarter'] = df_raw.index.quarter
    df_raw['Daily_Return'] = df_raw['Adjusted Close'].pct_change() * 100
    
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Monthly Returns Pattern', 'Quarterly Returns Pattern'))
    
    # Monthly patterns
    monthly_returns = df_raw.groupby('Month')['Daily_Return'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig.add_trace(go.Bar(x=months, y=monthly_returns.values, 
                        name='Avg Monthly Return', marker_color='skyblue'), row=1, col=1)
    
    # Quarterly patterns
    quarterly_returns = df_raw.groupby('Quarter')['Daily_Return'].mean()
    fig.add_trace(go.Bar(x=[f'Q{i}' for i in range(1,5)], y=quarterly_returns.values,
                        name='Avg Quarterly Return', marker_color='lightgreen'), row=1, col=2)
    
    fig.update_layout(title="📅 Best Months & Quarters to Invest", height=500)
    fig.update_yaxes(title_text="Average Return (%)")
    
    return fig

def create_bull_bear_analysis(df_raw):
    """5. Bull vs Bear market identification"""
    df_raw['MA_200'] = df_raw['Adjusted Close'].rolling(window=200).mean()
    df_raw['MA_50'] = df_raw['Adjusted Close'].rolling(window=50).mean()
    df_raw['Market_Trend'] = np.where(df_raw['Adjusted Close'] > df_raw['MA_200'], 'Bull Market', 'Bear Market')
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Adjusted Close'],
                            mode='lines', name='Stock Price', line=dict(color='black')))
    
    # Moving averages
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['MA_200'],
                            mode='lines', name='200-Day MA', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['MA_50'],
                            mode='lines', name='50-Day MA', line=dict(color='blue', dash='dash')))
    
    # Color background for bull/bear
    bull_periods = df_raw[df_raw['Market_Trend'] == 'Bull Market']
    for i in range(0, len(bull_periods), max(1, len(bull_periods)//10)):
        if i < len(bull_periods) - 1:
            fig.add_vrect(x0=bull_periods.index[i], x1=bull_periods.index[min(i+len(bull_periods)//10, len(bull_periods)-1)],
                         fillcolor="green", opacity=0.1, line_width=0)
    
    fig.update_layout(title="🐂🐻 Bull vs Bear Market Identification", 
                     xaxis_title="Date", yaxis_title="Price ($)", height=500)
    
    return fig

def create_risk_return_scatter(df_raw):
    """6. Risk-Return analysis by year"""
    df_raw['Year'] = df_raw.index.year
    df_raw['Daily_Return'] = df_raw['Adjusted Close'].pct_change()
    
    yearly_stats = df_raw.groupby('Year').agg({
        'Daily_Return': ['mean', 'std']
    }).round(4)
    
    yearly_stats.columns = ['Annual_Return', 'Risk']
    yearly_stats['Annual_Return'] *= 252 * 100  # Annualized return
    yearly_stats['Risk'] *= np.sqrt(252) * 100  # Annualized volatility
    yearly_stats = yearly_stats.reset_index()
    
    fig = px.scatter(yearly_stats, x='Risk', y='Annual_Return', 
                     text='Year', title="⚖️ Risk vs Return by Year - Investment Decision Guide")
    
    fig.update_traces(textposition="top center", marker=dict(size=10, color='blue'))
    fig.update_layout(xaxis_title="Risk (Volatility %)", yaxis_title="Annual Return (%)", height=500)
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=yearly_stats['Risk'].median(), line_dash="dash", line_color="gray")
    
    return fig

def create_volume_price_relationship(df_raw):
    """7. Volume-Price relationship"""
    df_raw['Price_Change'] = df_raw['Adjusted Close'].pct_change() * 100
    df_raw['Volume_MA'] = df_raw['Volume'].rolling(window=20).mean()
    df_raw['High_Volume'] = df_raw['Volume'] > df_raw['Volume_MA']
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=('Price Changes with Volume Signals', 'Volume Analysis'))
    
    # Price with volume signals
    colors = ['red' if x < 0 else 'green' for x in df_raw['Price_Change']]
    sizes = [8 if vol else 4 for vol in df_raw['High_Volume']]
    
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Price_Change'],
                            mode='markers', name='Price Changes',
                            marker=dict(color=colors, size=sizes)), row=1, col=1)
    
    # Volume bars
    fig.add_trace(go.Bar(x=df_raw.index, y=df_raw['Volume'], 
                        name='Volume', marker_color='lightblue'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Volume_MA'],
                            mode='lines', name='Volume MA', line=dict(color='orange')), row=2, col=1)
    
    fig.update_layout(title="📊 Volume Confirms Price Movements", height=600)
    fig.update_yaxes(title_text="Price Change (%)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_support_resistance(df_raw):
    """8. Support and Resistance levels"""
    # Calculate support and resistance using rolling min/max
    df_raw['Resistance'] = df_raw['High'].rolling(window=50, center=True).max()
    df_raw['Support'] = df_raw['Low'].rolling(window=50, center=True).min()
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df_raw.index[-500:],  # Last 500 days for clarity
                                open=df_raw['Open'][-500:],
                                high=df_raw['High'][-500:],
                                low=df_raw['Low'][-500:],
                                close=df_raw['Adjusted Close'][-500:],
                                name='Price'))
    
    # Support and resistance lines
    fig.add_trace(go.Scatter(x=df_raw.index[-500:], y=df_raw['Resistance'][-500:],
                            mode='lines', name='Resistance Level', 
                            line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=df_raw.index[-500:], y=df_raw['Support'][-500:],
                            mode='lines', name='Support Level', 
                            line=dict(color='green', dash='dot')))
    
    fig.update_layout(title="📈 Key Support & Resistance Levels - Entry/Exit Points", 
                     xaxis_title="Date", yaxis_title="Price ($)", height=500)
    
    return fig

def create_investment_timing(df_raw):
    """9. Best investment timing signals"""
    df_raw['MA_20'] = df_raw['Adjusted Close'].rolling(window=20).mean()
    df_raw['MA_50'] = df_raw['Adjusted Close'].rolling(window=50).mean()
    df_raw['RSI'] = calculate_rsi(df_raw['Adjusted Close'])
    
    # Generate buy/sell signals
    df_raw['Buy_Signal'] = ((df_raw['MA_20'] > df_raw['MA_50']) & 
                           (df_raw['MA_20'].shift(1) <= df_raw['MA_50'].shift(1)) &
                           (df_raw['RSI'] < 70))
    
    df_raw['Sell_Signal'] = ((df_raw['MA_20'] < df_raw['MA_50']) & 
                            (df_raw['MA_20'].shift(1) >= df_raw['MA_50'].shift(1)) &
                            (df_raw['RSI'] > 30))
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Adjusted Close'],
                            mode='lines', name='Stock Price', line=dict(color='black')))
    
    # Moving averages
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['MA_20'],
                            mode='lines', name='20-Day MA', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['MA_50'],
                            mode='lines', name='50-Day MA', line=dict(color='orange')))
    
    # Buy/Sell signals
    buy_signals = df_raw[df_raw['Buy_Signal']]
    sell_signals = df_raw[df_raw['Sell_Signal']]
    
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Adjusted Close'],
                            mode='markers', name='BUY Signal', 
                            marker=dict(color='green', size=10, symbol='triangle-up')))
    
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Adjusted Close'],
                            mode='markers', name='SELL Signal',
                            marker=dict(color='red', size=10, symbol='triangle-down')))
    
    fig.update_layout(title="⚡ Investment Timing Signals - When to Buy/Sell", 
                     xaxis_title="Date", yaxis_title="Price ($)", height=500)
    
    return fig

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# [Previous modeling functions remain the same - keeping them for space]
def train_model(X_train, X_test, y_train, y_test, model_name, model, resampling_method='auto', use_class_weights=True):
    """Train a single model with robust resampling and error handling"""
    try:
        balance_info = analyze_class_balance(y_train)
        
        if resampling_method != 'none' and balance_info['imbalance_ratio'] > 1.2:
            X_train_res, y_train_res, resampling_msg = robust_resampling(X_train, y_train, resampling_method)
        else:
            X_train_res, y_train_res = X_train, y_train
            resampling_msg = "No resampling applied"
        
        model_copy = None
        if use_class_weights and hasattr(model, 'class_weight'):
            class_weights = get_class_weights(y_train_res)
            if class_weights and balance_info['imbalance_ratio'] > 1.5:
                model_params = model.get_params()
                model_params['class_weight'] = class_weights
                model_copy = type(model)(**model_params)
            else:
                model_copy = model
        else:
            model_copy = model
        
        model_copy.fit(X_train_res, y_train_res)
        
        y_pred = model_copy.predict(X_test)
        y_pred_proba = None
        if hasattr(model_copy, 'predict_proba'):
            try:
                y_pred_proba = model_copy.predict_proba(X_test)[:, 1]
            except Exception:
                pass
        
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

# [Other modeling functions remain the same for brevity]

# Main app header
st.markdown('<h1 class="main-header">📈 Stock Market Analysis & Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎛️ Control Panel")
    st.markdown("---")
    
    st.subheader("📈 About This Tool")
    st.markdown("""
    **Complete Stock Analysis & Prediction** - Analyze historical data patterns and build ML models for investment decisions.
    
    🔄 **Process:** Explore data → Understand patterns → Train models → Make decisions
    """)

# Load data
if st.session_state.data is None or st.session_state.raw_data is None:
    with st.spinner("Loading datasets..."):
        st.session_state.data, st.session_state.raw_data = load_data()

if st.session_state.data is None or st.session_state.raw_data is None:
    st.stop()

df = st.session_state.data
df_raw = st.session_state.raw_data

# Main tabs - ADDED NEW ANALYSIS TAB
tab1, tab2, tab3, tab4 = st.tabs(["📊 Investment Analysis", "🔍 Data Explorer", "🤖 Model Lab", "📈 Performance Center"])

with tab1:
    st.header("📊 Investment Analysis - What the Data Tells You")
    st.markdown("### 💡 **For Laypeople: Understanding When and How to Invest**")
    
    # Create analysis plots
    with st.spinner("Generating investment analysis plots..."):
        
        # Plot 1: Price Trends
        st.subheader("1️⃣ Long-term Investment Overview")
        fig1 = create_price_trend_analysis(df_raw)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("**💰 Key Insight:** Look for overall upward trends and high volume periods for good entry points.")
        
        # Plot 2: Returns Distribution  
        st.subheader("2️⃣ Risk Assessment - Daily Returns")
        fig2 = create_return_distribution(df_raw)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("**⚠️ Key Insight:** Most daily returns are small. Large negative spikes indicate high-risk periods.")
        
        # Plot 3: Volatility
        st.subheader("3️⃣ Market Volatility - When to Be Cautious")
        fig3 = create_volatility_analysis(df_raw)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("**📊 Key Insight:** High volatility periods = higher risk. Conservative investors should avoid these times.")
        
        # Plot 4: Seasonal Patterns
        st.subheader("4️⃣ Best Times to Invest - Seasonal Patterns")
        fig4 = create_seasonal_patterns(df_raw)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("**📅 Key Insight:** Some months/quarters historically perform better. Use this for timing your investments.")
        
        # Plot 5: Bull vs Bear
        st.subheader("5️⃣ Bull vs Bear Markets - Market Trend Identification")
        fig5 = create_bull_bear_analysis(df_raw)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("**🐂🐻 Key Insight:** Price above 200-day average = Bull market (good time to invest). Below = Bear market (be cautious).")
        
        # Plot 6: Risk-Return
        st.subheader("6️⃣ Risk vs Return Analysis - Investment Decision Guide")
        fig6 = create_risk_return_scatter(df_raw)
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("**⚖️ Key Insight:** Top-right = High risk, high return. Bottom-left = Low risk, low return. Choose based on your tolerance.")
        
        # Plot 7: Volume-Price
        st.subheader("7️⃣ Volume Analysis - Confirming Price Movements")
        fig7 = create_volume_price_relationship(df_raw)
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("**📊 Key Insight:** Big price moves with high volume are more reliable. Large dots = high volume days.")
        
        # Plot 8: Support/Resistance
        st.subheader("8️⃣ Support & Resistance - Entry/Exit Points")
        fig8 = create_support_resistance(df_raw)
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("**📈 Key Insight:** Buy near support levels (green), sell near resistance levels (red). These are key price levels.")
        
        # Plot 9: Investment Timing
        st.subheader("9️⃣ Investment Timing Signals - When to Buy/Sell")
        fig9 = create_investment_timing(df_raw)
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown("**⚡ Key Insight:** Green triangles = Buy signals, Red triangles = Sell signals. Based on moving average crossovers.")
    
    # Summary for laypeople
    st.markdown("---")
    st.subheader("📋 Investment Summary for Beginners")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🛡️ Conservative Investor**
        - Wait for low volatility periods
        - Buy during bull markets (price above 200-day MA)
        - Focus on support levels for entry
        - Avoid high-risk months
        """)
    
    with col2:
        st.markdown("""
        **⚖️ Moderate Investor**
        - Use seasonal patterns for timing
        - Watch volume confirmations
        - Follow moving average signals
        - Balance risk-return based on goals
        """)
    
    with col3:
        st.markdown("""
        **🚀 Aggressive Investor**
        - High volatility = High opportunity
        - Use technical signals actively
        - Higher risk tolerance
        - Quick entry/exit on signals
        """)

with tab2:
    st.header("Data Explorer")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Dataset Overview")
        st.info(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.info(f"**Date Range:** {df.index.min().date()} to {df.index.max().date()}")
        
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
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=df.index, 
                y=df[selected_feature],
                mode='lines',
                name=selected_feature,
                line=dict(color='#1f77b4', width=2)
            ))
            fig_ts.update_layout(
                title=f"Time Series: {selected_feature}",
                xaxis_title="Date",
                yaxis_title=selected_feature,
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig_ts, use_container_width=True)
    
    # Distribution analysis
    st.subheader("Feature Distribution Analysis")
    if selected_feature:
        fig_dist = make_subplots(rows=1, cols=2, 
                               subplot_titles=(f'{selected_feature} Distribution', f'{selected_feature} by Movement'))
        
        # Histogram
        fig_dist.add_trace(go.Histogram(x=df[selected_feature], nbinsx=30, name='Distribution'), row=1, col=1)
        
        # Box plot by target
        for target in df['Target'].unique():
            data = df[df['Target'] == target][selected_feature]
            movement_type = 'Up Movement' if target == 1 else 'Down Movement'
            fig_dist.add_trace(go.Box(y=data, name=movement_type), row=1, col=2)
        
        fig_dist.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Feature Correlation Analysis")
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r"
    )
    fig_corr.update_layout(height=600, width=800)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.header("Model Lab")
    
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
            
            st.info(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
            
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
            
            total_models = len(selected_models)
            current_model = 0
            
            # Train models
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
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)

with tab4:
    st.header("Performance Center")
    
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
                    st.metric("🎯 Best F1-Score", f"{best_f1['f1_score']:.3f}", help=f"Model: {best_f1['name']}")
                with col2:
                    st.metric("🎯 Best Precision", f"{best_precision['precision']:.3f}", help=f"Model: {best_precision['name']}")
                with col3:
                    st.metric("🎯 Best Recall", f"{best_recall['recall']:.3f}", help=f"Model: {best_recall['name']}")
                with col4:
                    st.metric("📊 Models Trained", len(valid_results))
            except Exception as e:
                st.error(f"Error calculating best metrics: {str(e)}")
            
            # ROC Curves and Performance Comparison
            st.subheader("📈 Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC Curves
                try:
                    models_with_proba = {k: v for k, v in valid_results.items() if v['probabilities'] is not None}
                    if models_with_proba:
                        # Get test data
                        y_test = df.iloc[int(len(df) * 0.8):]['Target']
                        
                        fig_roc = go.Figure()
                        
                        for result in models_with_proba.values():
                            if result and result['probabilities'] is not None:
                                try:
                                    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                                    auc_score = result['auc_score']
                                    
                                    fig_roc.add_trace(go.Scatter(
                                        x=fpr, y=tpr,
                                        mode='lines',
                                        name=f"{result['name']} (AUC: {auc_score:.3f})",
                                        line=dict(width=2)
                                    ))
                                except Exception:
                                    continue
                        
                        # Add diagonal line
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random Classifier',
                            line=dict(dash='dash', color='gray')
                        ))
                        
                        fig_roc.update_layout(
                            title="ROC Curves Comparison",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            height=500,
                            template="plotly_white"
                        )
                        
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
                            'Model': result['name'][:15] + "..." if len(result['name']) > 15 else result['name'],
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
                        fig_metrics.update_layout(height=500)
                        st.plotly_chart(fig_metrics, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating metrics comparison: {str(e)}")
            
            # Individual model analysis
            st.subheader("🔍 Individual Model Analysis")
            
            model_names = list(valid_results.keys())
            selected_model_name = st.selectbox("Select Model for Detailed Analysis", model_names)
            
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
                    cm = selected_result['confusion_matrix']
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title=f"Confusion Matrix - {selected_result['name']}",
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Down Movement', 'Up Movement'],
                        y=['Down Movement', 'Up Movement'],
                        color_continuous_scale="Blues"
                    )
                    fig_cm.update_layout(height=400)
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with col2:
                    st.subheader("Model Insights")
                    st.markdown(f"**Resampling Strategy:** {selected_result['resampling_info']}")
                    
                    if 'class_balance' in selected_result:
                        balance_info = selected_result['class_balance']
                        st.markdown(f"**Original Imbalance Ratio:** {balance_info['imbalance_ratio']:.2f}")
                    
                    # Performance interpretation
                    f1 = selected_result['f1_score']
                    if f1 > 0.8:
                        st.success("🎉 Excellent model performance!")
                    elif f1 > 0.7:
                        st.info("✅ Good model performance")
                    elif f1 > 0.6:
                        st.warning("⚠️ Moderate performance - consider tuning")
                    else:
                        st.error("❌ Poor performance - needs improvement")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <h4>📈 Complete Stock Market Analysis & Prediction Dashboard</h4>
        <p><strong>Features:</strong> Investment Analysis for Laypeople • Advanced ML Models • Robust Data Handling</p>
        <p><strong>Built with:</strong> Streamlit • Scikit-learn • Plotly • Technical Analysis</p>
        <p><em>Made by Ahmed Awad</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)
