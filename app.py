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
    page_title="Stock Market Prediction Dashboard",
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

@st.cache_data
def load_data():
    """Load and return the preprocessed dataset"""
    try:
        df = pd.read_csv("ENB_data_binary_classification.csv", index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'ENB_data_binary_classification.csv' is in the working directory.")
        return None

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

def create_time_series_plot(df, feature):
    """Create time series plot for selected feature"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df[feature],
        mode='lines',
        name=feature,
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=f"Time Series: {feature}",
        xaxis_title="Date",
        yaxis_title=feature,
        height=400,
        template="plotly_white"
    )
    return fig

def create_distribution_plot(df, feature):
    """Create distribution plot for selected feature"""
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=(f'{feature} Distribution', f'{feature} by Movement'))
    
    # Histogram
    fig.add_trace(go.Histogram(x=df[feature], nbinsx=30, name='Distribution'), row=1, col=1)
    
    # Box plot by target
    for target in df['Target'].unique():
        data = df[df['Target'] == target][feature]
        movement_type = 'Up Movement' if target == 1 else 'Down Movement'
        fig.add_trace(go.Box(y=data, name=movement_type), row=1, col=2)
    
    fig.update_layout(height=400, template="plotly_white")
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
                    class_weight='balanced'  # Always use balanced class weights in AutoML
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
st.markdown('<h1 class="main-header">📈 Stock Market Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎛️ Control Panel")
    st.markdown("---")
    
    # Tool Description
    st.subheader("📈 About This Tool")
    st.markdown("""
    **Stock Market Prediction Dashboard** helps you build and compare machine learning models for predicting stock price movements.
    
    🔄 **Process:** Upload data → Explore features → Train models → Analyze performance
    """)
    
    st.markdown("---")
    
    # Use Cases
    st.subheader("💼 Key Use Cases")
    
    with st.expander("🛡️ Conservative Investor", expanded=True):
        st.markdown("""
        **Focus: High Precision**
        
        • Minimize false positives (avoid bad investments)
        • Prefer models with 80%+ precision
        • Better to miss opportunities than lose money
        • Use SVM or Logistic Regression for interpretability
        """)
    
    with st.expander("🚀 Growth Investor"):
        st.markdown("""
        **Focus: High Recall**
        
        • Capture most growth opportunities
        • Don't mind some false positives
        • Prefer models with 75%+ recall
        • Use Random Forest or Gradient Boosting
        """)
    
    with st.expander("⚖️ Balanced Trader"):
        st.markdown("""
        **Focus: High F1-Score**
        
        • Balance between precision and recall
        • Optimize overall model performance
        • Use AutoML for best parameter tuning
        • Compare multiple models systematically
        """)
    
    st.markdown("---")
    st.markdown("*💡 Choose your strategy based on your risk tolerance and investment goals.*")

# Load data
if st.session_state.data is None:
    with st.spinner("Loading dataset..."):
        st.session_state.data = load_data()

if st.session_state.data is None:
    st.stop()

df = st.session_state.data

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🤖 Model Lab", "📈 Performance Center"])

with tab1:
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
            fig_ts = create_time_series_plot(df, selected_feature)
            st.plotly_chart(fig_ts, use_container_width=True)
    
    # Distribution analysis
    st.subheader("Feature Distribution Analysis")
    if selected_feature:
        fig_dist = create_distribution_plot(df, selected_feature)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Feature Correlation Analysis")
    fig_corr = create_correlation_heatmap(df)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.header("Model Lab")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Model Configuration")
        
        # Feature selection
        feature_cols = [col for col in df.columns if col != 'Target']
        selected_features = st.multiselect(
            "Select Features",
            feature_cols,
            default=feature_cols[:5] if len(feature_cols) > 5 else feature_cols  # Limit default selection
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
        
        # Model selection with better defaults
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
            use_class_weights = st.checkbox("Use Class Weights", value=True, 
                                          help="Automatically balance classes using sample weights")
            
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
                    ["f1", "precision", "recall", "accuracy"],
                    help="Metric to optimize during hyperparameter search"
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
                X = X.fillna(X.mean())  # Simple imputation
            
            # Train-test split (chronological)
            split_index = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            # Display data split info
            st.info(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
            
            # Analyze class balance in training set
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
                        'AUC': f"{result['auc_score']:.3f}" if result['auc_score'] else "N/A",
                        'Resampling': result['resampling_info'][:30] + "..." if len(result['resampling_info']) > 30 else result['resampling_info']
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Show best performing model
                if results_data:
                    best_f1_model = max(st.session_state.model_results.values(), key=lambda x: x['f1_score'])
                    st.success(f"🏆 Best F1-Score: {best_f1_model['name']} ({best_f1_model['f1_score']:.3f})")
            else:
                st.warning("⚠️ No models were successfully trained. Please check your data and configuration.")
        
        elif not st.session_state.models_trained:
            st.info("👈 Configure your models and click 'Train Models' to see results here.")
            
            # Show data preview
            if not df.empty:
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)

with tab3:
    st.header("Performance Center")
    
    if not st.session_state.model_results:
        st.warning("⚠️ No models trained yet. Please go to the Model Lab tab to train some models first.")
    else:
        # Model comparison metrics
        st.subheader("📊 Model Comparison Dashboard")
        
        # Get all valid results (filter out None results)
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
                    st.metric(
                        "🎯 Best F1-Score", 
                        f"{best_f1['f1_score']:.3f}", 
                        help=f"Model: {best_f1['name']}"
                    )
                with col2:
                    st.metric(
                        "🎯 Best Precision", 
                        f"{best_precision['precision']:.3f}",
                        help=f"Model: {best_precision['name']}"
                    )
                with col3:
                    st.metric(
                        "🎯 Best Recall", 
                        f"{best_recall['recall']:.3f}",
                        help=f"Model: {best_recall['name']}"
                    )
                with col4:
                    st.metric("📊 Models Trained", len(valid_results))
            except Exception as e:
                st.error(f"Error calculating best metrics: {str(e)}")
            
            # Detailed comparison charts
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
            
            # Class imbalance analysis
            st.subheader("⚖️ Class Balance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Show original class distribution
                target_dist = df['Target'].value_counts().sort_index()
                fig_class_dist = px.pie(
                    values=target_dist.values,
                    names=['Down Movement', 'Up Movement'],
                    title="Original Class Distribution",
                    color_discrete_sequence=['#ff6b6b', '#4ecdc4']
                )
                st.plotly_chart(fig_class_dist, use_container_width=True)
            
            with col2:
                # Show resampling summary
                st.subheader("Resampling Summary")
                resampling_summary = []
                for model_name, result in valid_results.items():
                    if result and 'resampling_info' in result:
                        resampling_summary.append({
                            'Model': model_name,
                            'Strategy': result['resampling_info']
                        })
                
                if resampling_summary:
                    resampling_df = pd.DataFrame(resampling_summary)
                    st.dataframe(resampling_df, use_container_width=True)
                else:
                    st.info("No resampling information available")
            
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
                
                # Cache expensive computations
                @st.cache_data
                def get_model_visualizations(model_name, cm_data, feature_names, model_type):
                    """Cache model visualizations to improve performance"""
                    try:
                        # Confusion Matrix
                        cm_fig = create_confusion_matrix_plot(cm_data, model_name)
                        
                        # Feature Importance (if available)
                        fi_fig = None
                        if feature_names and len(feature_names) > 0:
                            # Check if we can get feature importance
                            if hasattr(selected_result['model'], 'feature_importances_'):
                                importances = selected_result['model'].feature_importances_
                                fi_fig = create_feature_importance_plot(selected_result, feature_names)
                            elif hasattr(selected_result['model'], 'coef_'):
                                importances = np.abs(selected_result['model'].coef_[0])
                                fi_fig = create_feature_importance_plot(selected_result, feature_names)
                        
                        return cm_fig, fi_fig
                    except Exception as e:
                        st.error(f"Error creating visualizations: {str(e)}")
                        return None, None
                
                # Model details in expandable sections
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
                        
                        # Class balance info
                        if 'class_balance' in selected_result:
                            balance_info = selected_result['class_balance']
                            st.metric("Imbalance Ratio", f"{balance_info['imbalance_ratio']:.2f}")
                
                # Get cached visualizations
                feature_names_for_viz = selected_features if 'selected_features' in locals() else []
                model_type = type(selected_result['model']).__name__
                
                cm_fig, fi_fig = get_model_visualizations(
                    selected_result['name'],
                    selected_result['confusion_matrix'],
                    feature_names_for_viz,
                    model_type
                )
                
                # Display visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    if cm_fig:
                        st.plotly_chart(cm_fig, use_container_width=True)
                    else:
                        st.error("Could not generate confusion matrix")
                
                with col2:
                    if fi_fig:
                        st.plotly_chart(fi_fig, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model type.")
                
                # Classification Report
                with st.expander("📊 Detailed Classification Report"):
                    try:
                        if 'classification_report' in selected_result:
                            # Use cached data processing
                            @st.cache_data
                            def process_classification_report(report_data):
                                report_df = pd.DataFrame(report_data).transpose()
                                
                                # Format numeric columns
                                for col in ['precision', 'recall', 'f1-score']:
                                    if col in report_df.columns:
                                        report_df[col] = report_df[col].apply(
                                            lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else str(x)
                                        )
                                return report_df
                            
                            report_df = process_classification_report(selected_result['classification_report'])
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
                            st.metric("Trials Run", f"{n_trials if 'n_trials' in locals() else 'N/A'}")
                            st.text(f"• Optimized for: {automl_metric.upper() if 'automl_metric' in locals() else 'F1'}")


# Model Parameters Section
st.markdown("---")
st.subheader("🔧 Model Parameters & Configuration")

if st.session_state.model_results:
    valid_results = {k: v for k, v in st.session_state.model_results.items() if v is not None}
    
    if valid_results:
        # Create tabs for different parameter views
        param_tab1, param_tab2 = st.tabs(["📋 Model Parameters", "🎯 AutoML Results"])
        
        with param_tab1:
            st.subheader("Standard Model Parameters")
            
            for model_name, result in valid_results.items():
                if 'AutoML' not in model_name:  # Show only non-AutoML models
                    with st.expander(f"⚙️ {model_name} Parameters"):
                        model = result['model']
                        params = model.get_params()
                        
                        # Display key parameters in columns
                        param_cols = st.columns(3)
                        param_count = 0
                        
                        for param, value in params.items():
                            if param_count < 9:  # Show first 9 parameters
                                with param_cols[param_count % 3]:
                                    st.metric(param, str(value)[:20])
                                param_count += 1
        
        with param_tab2:
            st.subheader("AutoML Optimization Results")
            
            automl_found = False
            for model_name, result in valid_results.items():
                if 'AutoML' in model_name and 'automl_params' in result:
                    automl_found = True
                    with st.expander(f"🚀 {model_name} - Best Parameters"):
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Optimized Parameters:**")
                            for param, value in result['automl_params'].items():
                                st.text(f"• {param}: {value}")
                        
                        with col2:
                            st.write("**Performance:**")
                            st.metric("Best Score", f"{result['automl_best_score']:.3f}")
                            st.metric("Final F1-Score", f"{result['f1_score']:.3f}")
                            st.metric("Final Accuracy", f"{result['accuracy']:.3f}")
            
            if not automl_found:
                st.info("No AutoML models found. Enable AutoML in the Model Lab to see optimization results.")
    
else:
    st.info("Train some models first to see their parameters and configurations.")

# Footer with enhanced information
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <h4>🚀 Enhanced Stock Market Prediction Dashboard</h4>
        <p><strong>Features:</strong> Robust SMOTE handling • Multiple fallback strategies • Class weight balancing • AutoML optimization</p>
        <p><strong>Built with:</strong> Streamlit • Scikit-learn • Plotly • Optuna</p>
        <p><em>Made by Ahmed Awad</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)
