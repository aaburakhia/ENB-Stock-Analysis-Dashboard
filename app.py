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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
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
                       subplot_titles=(f'{feature} Distribution', f'{feature} by Target'))
    
    # Histogram
    fig.add_trace(go.Histogram(x=df[feature], nbinsx=30, name='Distribution'), row=1, col=1)
    
    # Box plot by target
    for target in df['Target'].unique():
        data = df[df['Target'] == target][feature]
        fig.add_trace(go.Box(y=data, name=f'Target {target}'), row=1, col=2)
    
    fig.update_layout(height=400, template="plotly_white")
    return fig

def train_model(X_train, X_test, y_train, y_test, model_name, model, use_smote=True):
    """Train a single model and return results"""
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
    
    # Train model
    model.fit(X_train_res, y_train_res)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    return {
        'model': model,
        'name': model_name,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_score': auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def run_automl_optimization(X_train, X_test, y_train, y_test, model_type, n_trials=50):
    """Run AutoML optimization using Optuna"""
    
    def objective(trial):
        if model_type == "Random Forest":
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
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
            model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=42)
        else:  # Logistic Regression
            C = trial.suggest_float('C', 0.01, 10)
            model = LogisticRegression(C=C, random_state=42)
        
        # Use SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Train best model
    best_params = study.best_params
    
    if model_type == "Random Forest":
        best_model = RandomForestClassifier(**best_params, random_state=42)
    elif model_type == "Gradient Boosting":
        best_model = GradientBoostingClassifier(**best_params, random_state=42)
    elif model_type == "SVM":
        best_model = SVC(**best_params, probability=True, random_state=42)
    else:
        best_model = LogisticRegression(**best_params, random_state=42)
    
    return train_model(X_train, X_test, y_train, y_test, f"{model_type} (AutoML)", best_model)

def create_roc_curve(model_results):
    """Create ROC curve for multiple models"""
    fig = go.Figure()
    
    # Get test data for ROC curve calculation
    df = st.session_state.data
    feature_cols = [col for col in df.columns if col != 'Target']
    y_test = df.iloc[int(len(df) * 0.8):]['Target']
    
    for result in model_results.values():
        if result['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            auc_score = result['auc_score']
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"{result['name']} (AUC: {auc_score:.3f})",
                line=dict(width=2)
            ))
    
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
        x=['Negative', 'Positive'],
        y=['Negative', 'Positive'],
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
    
    # Data loading status
    if st.button("🔄 Reload Data", type="primary"):
        st.session_state.data = None
        st.session_state.models_trained = False
        st.session_state.model_results = {}
        st.rerun()

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
        
        # Class distribution
        target_dist = df['Target'].value_counts()
        st.metric("Positive Cases", target_dist.get(1, 0))
        st.metric("Negative Cases", target_dist.get(0, 0))
        st.metric("Class Balance", f"{target_dist.get(1, 0) / len(df):.1%}")
    
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
            default=feature_cols
        )
        
        # Model selection
        available_models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Naive Bayes": GaussianNB(),
            "K-Nearest Neighbors": KNeighborsClassifier()
        }
        
        selected_models = st.multiselect(
            "Select Models",
            list(available_models.keys()),
            default=["Logistic Regression", "Random Forest"]
        )
        
        # AutoML option
        use_automl = st.checkbox("🚀 Use AutoML Optimization")
        if use_automl:
            n_trials = st.slider("AutoML Trials", 10, 100, 50)
            automl_model = st.selectbox(
                "Model for AutoML",
                ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression"]
            )
        
        # Training options
        use_smote = st.checkbox("Use SMOTE for Class Balancing", value=True)
        
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
            
            # Train-test split (chronological)
            split_index = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_models = len(selected_models) + (1 if use_automl else 0)
            
            # Train regular models
            for i, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}...")
                model = available_models[model_name]
                result = train_model(X_train_scaled, X_test_scaled, y_train, y_test, model_name, model, use_smote)
                st.session_state.model_results[model_name] = result
                progress_bar.progress((i + 1) / total_models)
            
            # Train AutoML model if selected
            if use_automl:
                status_text.text(f"Running AutoML optimization for {automl_model}...")
                automl_result = run_automl_optimization(X_train_scaled, X_test_scaled, y_train, y_test, automl_model, n_trials)
                st.session_state.model_results[f"{automl_model} (AutoML)"] = automl_result
                progress_bar.progress(1.0)
            
            status_text.text("Training completed!")
            
            # Display quick results
            st.subheader("Quick Results")
            results_df = pd.DataFrame({
                'Model': [result['name'] for result in st.session_state.model_results.values()],
                'Accuracy': [f"{result['accuracy']:.3f}" for result in st.session_state.model_results.values()],
                'F1-Score': [f"{result['f1_score']:.3f}" for result in st.session_state.model_results.values()],
                'AUC': [f"{result['auc_score']:.3f}" if result['auc_score'] else "N/A" for result in st.session_state.model_results.values()]
            })
            
            st.dataframe(results_df, use_container_width=True)
        
        elif not st.session_state.models_trained:
            st.info("👈 Configure your models and click 'Train Models' to see results here.")

with tab3:
    st.header("Performance Center")
    
    if not st.session_state.model_results:
        st.warning("⚠️ No models trained yet. Please go to the Model Lab tab to train some models first.")
    else:
        # Model comparison metrics
        st.subheader("📊 Model Comparison")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Best model metrics
        best_f1 = max(st.session_state.model_results.values(), key=lambda x: x['f1_score'])
        best_acc = max(st.session_state.model_results.values(), key=lambda x: x['accuracy'])
        best_auc = max([r for r in st.session_state.model_results.values() if r['auc_score']], 
                       key=lambda x: x['auc_score'], default=best_f1)
        
        with col1:
            st.metric("🎯 Best F1-Score", f"{best_f1['f1_score']:.3f}", best_f1['name'])
        with col2:
            st.metric("🎯 Best Accuracy", f"{best_acc['accuracy']:.3f}", best_acc['name'])
        with col3:
            st.metric("🎯 Best AUC", f"{best_auc['auc_score']:.3f}" if best_auc['auc_score'] else "N/A", best_auc['name'])
        with col4:
            st.metric("📊 Models Trained", len(st.session_state.model_results), "")
        
        # Detailed comparison charts
        st.subheader("📈 Detailed Performance Analysis")
        
        # ROC Curves
        col1, col2 = st.columns(2)
        
        with col1:
            if any(r['auc_score'] for r in st.session_state.model_results.values()):
                fig_roc = create_roc_curve(st.session_state.model_results)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC curves require models with probability predictions.")
        
        with col2:
            # Metrics comparison
            metrics_data = []
            for result in st.session_state.model_results.values():
                metrics_data.append({
                    'Model': result['name'],
                    'Accuracy': result['accuracy'],
                    'F1-Score': result['f1_score'],
                    'AUC': result['auc_score'] or 0
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            fig_metrics = px.bar(
                metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Model', y='Score', color='Metric',
                title="Performance Metrics Comparison",
                barmode='group'
            )
            fig_metrics.update_xaxes(tickangle=45)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Individual model analysis
        st.subheader("🔍 Individual Model Analysis")
        
        selected_model_name = st.selectbox(
            "Select Model for Detailed Analysis",
            list(st.session_state.model_results.keys())
        )
        
        if selected_model_name:
            selected_result = st.session_state.model_results[selected_model_name]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                fig_cm = create_confusion_matrix_plot(
                    selected_result['confusion_matrix'],
                    selected_result['name']
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                # Feature Importance
                if len(selected_features) > 0:
                    fig_fi = create_feature_importance_plot(selected_result, selected_features)
                    if fig_fi:
                        st.plotly_chart(fig_fi, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model type.")
                else:
                    st.info("Please train models first to see feature importance.")
            
            # Classification Report
            st.subheader("📋 Detailed Classification Report")
            
            if 'classification_report' in selected_result:
                report_df = pd.DataFrame(selected_result['classification_report']).transpose()
                # Format numeric columns
                for col in ['precision', 'recall', 'f1-score']:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
                
                st.dataframe(report_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "🚀 **Stock Market Prediction Dashboard** | Built with Streamlit | "
    "University Capstone Project 2025"
)
