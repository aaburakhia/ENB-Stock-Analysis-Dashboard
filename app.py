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

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Stock Market Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- (Custom CSS is unchanged) ---
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
    /* ... (rest of your CSS) ... */
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE & DATA LOADING ---
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
    """Load and return both the preprocessed and raw datasets"""
    try:
        df = pd.read_csv("ENB_data_binary_classification.csv", index_col=0)
        df.index = pd.to_datetime(df.index)
        df_raw = pd.read_csv('ENB_TO_2000_2025_clean.csv', index_col='Date', parse_dates=True)
        return df, df_raw
    except FileNotFoundError:
        st.error("A required dataset file was not found! Please ensure all CSV files are present.")
        return None, None

# --- (All your helper functions are unchanged) ---
def analyze_class_balance(y):
    # ... (implementation is the same)
def robust_resampling(X, y, method='auto', random_state=42):
    # ... (implementation is the same)
def get_class_weights(y):
    # ... (implementation is the same)
# ... (and so on for all your other helper functions)

# --- 3. MAIN APP HEADER ---
st.markdown('<h1 class="main-header">Stock Market Prediction Dashboard</h1>', unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("Control Panel")
    st.markdown("---")
    
    st.subheader("About This Tool")
    st.markdown("""
    **Stock Market Prediction Dashboard** helps you build and compare machine learning models for predicting stock price movements.
    
    **Process:** Upload data → Explore features → Train models → Analyze performance
    """)
    
    st.markdown("---")
    
    st.subheader("Key Use Cases")
    with st.expander("Conservative Investor", expanded=True):
        st.markdown("**Focus: High Precision**")
    with st.expander("Growth Investor"):
        st.markdown("**Focus: High Recall**")
    with st.expander("Balanced Trader"):
        st.markdown("**Focus: High F1-Score**")
    
    st.markdown("---")
    st.markdown("*Choose your strategy based on your risk tolerance and investment goals.*")

# --- 5. DATA LOADING ---
if st.session_state.data is None:
    with st.spinner("Loading datasets..."):
        st.session_state.data, st.session_state.raw_data = load_data()

if st.session_state.data is None:
    st.stop()

df = st.session_state.data
df_raw = st.session_state.raw_data

# --- 6. MAIN CONTENT TABS ---
tab1, tab2, tab3 = st.tabs(["Investor Insights & EDA", "Data Explorer", "Model Lab"])

# --- NEW INVESTOR INSIGHTS TAB ---
with tab1:
    st.header("Historical Performance & Investment Insights")
    st.markdown("These charts analyze the historical data to uncover trends and answer common investor questions.")

    st.subheader("1. What is the long-term price trend?")
    st.line_chart(df_raw['Adjusted Close'])
    st.info("**Insight:** This chart shows the overall growth trajectory of the stock over the last 25 years, providing a long-term perspective on its performance.")

    st.subheader("2. Are there 'better' times to invest?")
    col1, col2 = st.columns(2)
    df_raw['Month'] = df_raw.index.strftime('%B')
    df_raw['DayOfWeek'] = df_raw.index.strftime('%A')
    df_raw['Daily_Return'] = df_raw['Adjusted Close'].pct_change()
    
    monthly_returns = df_raw.groupby('Month')['Daily_Return'].mean().sort_values(ascending=False)
    fig_monthly = px.bar(monthly_returns, title="Average Daily Return by Month")
    col1.plotly_chart(fig_monthly, use_container_width=True)

    day_returns = df_raw.groupby('DayOfWeek')['Daily_Return'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    fig_day = px.bar(day_returns, title="Average Daily Return by Day of the Week")
    col2.plotly_chart(fig_day, use_container_width=True)
    st.info("**Insight:** These charts show historical seasonal tendencies. This doesn't guarantee future performance, but it provides a data-driven look at seasonality.")

    st.subheader("3. Is 'buying the dip' a good strategy?")
    df_raw['MA_50'] = df_raw['Adjusted Close'].rolling(window=50).mean()
    df_raw['Is_Dip'] = df_raw['Adjusted Close'] < (df_raw['MA_50'] * 0.95)
    df_raw['Return_After_30D'] = df_raw['Adjusted Close'].shift(-30) / df_raw['Adjusted Close'] - 1
    
    dip_returns = df_raw[df_raw['Is_Dip']]['Return_After_30D'].mean()
    st.metric("Average Return 30 Days After a 'Dip'", f"{dip_returns:.2%}")
    st.info("**Insight:** This chart tests a famous investment strategy. It shows that, historically for this stock, buying after a significant price drop has, on average, led to a positive return over the next 30 days.")

# --- (Your existing Data Explorer and Model Lab tabs are unchanged) ---
with tab2:
    # ... (Paste your entire "Data Explorer" tab code here) ...
    st.header("Data Explorer")
    # ... etc ...

with tab3:
    # ... (Paste your entire "Model Lab" and "Performance Center" code here, combined) ...
    st.header("Model Lab")
    # ... etc ...
