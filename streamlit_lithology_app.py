"""
🪨 Lithology Classification ML Pipeline - Streamlit App
======================================================

Interactive web application for lithology classification with
comprehensive data quality tracking and detailed annotations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Lithology ML Pipeline",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_sample_data():
    """Load and combine sample well log data"""
    try:
        # Try to load existing data files
        data_files = []
        data_dir = "litho_data"

        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(data_dir, file))
                        if 'FORCE_2020_LITHOFACIES_LITHOLOGY' in df.columns:
                            data_files.append(df)
                    except:
                        continue

        if data_files:
            combined_data = pd.concat(data_files, ignore_index=True)
            return combined_data
        else:
            # Create sample data if no files found
            return create_sample_data()

    except Exception as e:
        return create_sample_data()

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'imputer' not in st.session_state:
    st.session_state.imputer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = None
if 'classification_report_data' not in st.session_state:
    st.session_state.classification_report_data = None

@st.cache_data
def create_sample_data():
    """Create sample well log data with multiple lithology types"""
    np.random.seed(42)
    n_samples = 15000  # Increased sample size for better training

    # Define lithology types with more realistic and distinct characteristics
    lithologies = {
        'Sandstone': {'GR': (15, 75), 'RHOB': (1.95, 2.45), 'NPHI': (0.03, 0.28), 'RDEP': (8, 1200), 'DTC': (55, 105), 'PEF': (1.6, 2.9)},
        'Shale': {'GR': (75, 220), 'RHOB': (2.15, 2.85), 'NPHI': (0.12, 0.48), 'RDEP': (0.8, 25), 'DTC': (75, 150), 'PEF': (2.7, 3.6)},
        'Limestone': {'GR': (8, 65), 'RHOB': (2.35, 2.85), 'NPHI': (-0.02, 0.18), 'RDEP': (40, 6000), 'DTC': (45, 85), 'PEF': (4.2, 5.8)},
        'Dolomite': {'GR': (5, 55), 'RHOB': (2.55, 2.95), 'NPHI': (-0.08, 0.12), 'RDEP': (80, 12000), 'DTC': (40, 80), 'PEF': (2.6, 3.4)},
        'Anhydrite': {'GR': (2, 35), 'RHOB': (2.75, 3.05), 'NPHI': (-0.12, 0.08), 'RDEP': (800, 60000), 'DTC': (45, 75), 'PEF': (4.8, 6.2)},
        'Salt': {'GR': (0, 25), 'RHOB': (1.95, 2.25), 'NPHI': (-0.12, 0.02), 'RDEP': (8000, 120000), 'DTC': (60, 90), 'PEF': (4.2, 5.8)},
        'Coal': {'GR': (40, 180), 'RHOB': (1.1, 1.7), 'NPHI': (0.25, 0.85), 'RDEP': (80, 15000), 'DTC': (90, 220), 'PEF': (0.1, 0.9)},
        'Marl': {'GR': (55, 130), 'RHOB': (2.25, 2.65), 'NPHI': (0.08, 0.32), 'RDEP': (3, 120), 'DTC': (65, 115), 'PEF': (3.2, 4.8)},
        'Sandstone_Shaly': {'GR': (55, 130), 'RHOB': (2.05, 2.55), 'NPHI': (0.08, 0.32), 'RDEP': (3, 250), 'DTC': (65, 125), 'PEF': (2.0, 3.4)},
        'Limestone_Shaly': {'GR': (35, 110), 'RHOB': (2.25, 2.75), 'NPHI': (0.03, 0.28), 'RDEP': (8, 600), 'DTC': (55, 105), 'PEF': (3.6, 5.0)},
        'Siltstone': {'GR': (45, 95), 'RHOB': (2.1, 2.5), 'NPHI': (0.05, 0.25), 'RDEP': (5, 300), 'DTC': (65, 110), 'PEF': (2.2, 3.0)},
        'Mudstone': {'GR': (70, 160), 'RHOB': (2.2, 2.7), 'NPHI': (0.15, 0.40), 'RDEP': (2, 50), 'DTC': (80, 130), 'PEF': (2.8, 3.4)}
    }

    data = []
    samples_per_lithology = n_samples // len(lithologies)

    for i, (lith_name, params) in enumerate(lithologies.items()):
        for j in range(samples_per_lithology):
            row = {
                'WELL': f'SAMPLE_WELL_{(i*samples_per_lithology + j) // 500 + 1}',
                'DEPTH_MD': 2000 + (i*samples_per_lithology + j) * 0.5,
                'FORCE_2020_LITHOFACIES_LITHOLOGY': lith_name
            }

            # Generate realistic log values with some noise
            for param, (min_val, max_val) in params.items():
                base_val = np.random.uniform(min_val, max_val)
                noise = np.random.normal(0, (max_val - min_val) * 0.05)
                row[param] = max(0, base_val + noise)

            data.append(row)

    df = pd.DataFrame(data)

    # Add some missing values (5% of data)
    for col in ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF']:
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, col] = np.nan

    return df

def analyze_data_quality(data):
    """Analyze data quality and return summary"""
    quality_summary = {}

    # Check for missing values
    missing_data = data.isnull().sum()
    quality_summary['missing_values'] = missing_data[missing_data > 0].to_dict()

    # Check data types
    quality_summary['data_types'] = data.dtypes.to_dict()

    # Basic statistics
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    quality_summary['statistics'] = data[numeric_cols].describe().to_dict()

    return quality_summary

def preprocess_data(data):
    """Preprocess data for machine learning"""
    # Define feature columns
    feature_columns = ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF']
    target_column = 'FORCE_2020_LITHOFACIES_LITHOLOGY'

    # Filter available features
    available_features = [col for col in feature_columns if col in data.columns]

    if len(available_features) < 3:
        st.error(f"Insufficient features. Need at least 3, got {len(available_features)}")
        return None, None, None, None, None

    # Remove rows with missing target
    clean_data = data.dropna(subset=[target_column]).copy()

    # Extract features and target
    X = clean_data[available_features].copy()
    y = clean_data[target_column].copy()

    # Handle missing values in features
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=available_features,
        index=X.index
    )

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=available_features,
        index=X_imputed.index
    )

    # Clean target labels - ensure all are strings
    y_clean = y.astype(str)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_clean)

    return X_scaled, y_encoded, label_encoder, imputer, scaler

@st.cache_resource
def train_model(X, y, model_type='random_forest'):
    """Train optimized machine learning model"""
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == 'random_forest':
        # Optimized Random Forest with better hyperparameters
        model = RandomForestClassifier(
            n_estimators=200,  # More trees for better performance
            max_depth=20,      # Deeper trees for complex patterns
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
    elif model_type == 'extra_trees':
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, y_pred_proba, accuracy

def create_quality_annotations(original_data, processed_data, imputer, scaler):
    """Create quality annotations for the dataset"""
    annotations = pd.DataFrame(index=original_data.index)

    # Check for null values
    feature_columns = ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF']
    available_features = [col for col in feature_columns if col in original_data.columns]

    annotations['Has_Null_Values'] = original_data[available_features].isnull().any(axis=1)
    annotations['Null_Count'] = original_data[available_features].isnull().sum(axis=1)

    # Quality issues description
    quality_issues = []
    for idx in annotations.index:
        issues = []
        for col in available_features:
            if pd.isna(original_data.loc[idx, col]):
                issues.append(f"Null in {col}")
        annotations.loc[idx, 'Quality_Issues'] = '; '.join(issues) if issues else 'None'

    # Processing techniques applied
    annotations['Techniques_Applied'] = 'Median Imputation; StandardScaler'
    annotations['Data_Quality_Score'] = 100 - (annotations['Null_Count'] * 10)  # Simple scoring
    annotations['Data_Quality_Score'] = annotations['Data_Quality_Score'].clip(lower=0)

    return annotations

def create_processed_dataset_with_annotations(original_data, processed_features, quality_annotations, predictions=None, confidence=None):
    """Create processed dataset with detailed annotations for download"""

    # Start with original data
    processed_dataset = original_data.copy()

    # Add processed feature values (scaled)
    feature_columns = ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF']
    available_features = [col for col in feature_columns if col in processed_features.columns]

    # Add processed (scaled) values with prefix
    for col in available_features:
        if col in processed_features.columns:
            processed_dataset[f'{col}_Processed'] = processed_features[col].values

    # Detect outliers in original data
    for col in available_features:
        if col in original_data.columns:
            # IQR method for outlier detection
            Q1 = original_data[col].quantile(0.25)
            Q3 = original_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Mark outliers
            outlier_mask = (original_data[col] < lower_bound) | (original_data[col] > upper_bound)
            processed_dataset[f'{col}_Is_Outlier'] = outlier_mask
            processed_dataset[f'{col}_Outlier_Bounds'] = processed_dataset.apply(
                lambda row: f"[{lower_bound:.2f}, {upper_bound:.2f}]" if row[f'{col}_Is_Outlier'] else "Normal", axis=1
            )

    # Add quality annotations
    if not quality_annotations.empty:
        for col in quality_annotations.columns:
            processed_dataset[f'Quality_{col}'] = quality_annotations[col]

    # Add detailed processing information
    processed_dataset['Processing_Method'] = 'Median_Imputation_StandardScaler'
    processed_dataset['Outlier_Treatment'] = 'Preserved_For_Geological_Validity'
    processed_dataset['Missing_Value_Treatment'] = processed_dataset.apply(
        lambda row: 'Median_Imputation_Applied' if row.get('Quality_Has_Null_Values', False) else 'No_Treatment_Needed', axis=1
    )

    # Add model predictions if available
    if predictions is not None:
        processed_dataset['ML_Predicted_Lithology'] = predictions
        if confidence is not None:
            processed_dataset['Prediction_Confidence'] = confidence
            processed_dataset['Confidence_Category'] = pd.cut(
                confidence,
                bins=[0, 0.6, 0.8, 1.0],
                labels=['Low_Confidence', 'Medium_Confidence', 'High_Confidence']
            )

    # Add processing metadata
    processed_dataset['Processing_Timestamp'] = pd.Timestamp.now()
    processed_dataset['Dataset_Version'] = 'Processed_With_Quality_Tracking_v1.0'

    return processed_dataset

def main():
    """Main Streamlit application"""

    # Header
    st.title("🪨 Lithology Classification ML Pipeline")
    st.markdown("Predict lithology types from well log data with comprehensive quality tracking")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Data upload option
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file (optional)",
            type=['csv'],
            help="Upload your own well log data or use sample dataset"
        )

        # Processing options
        st.subheader("🎯 Processing Options")
        include_quality_tracking = st.checkbox(
            "Quality Tracking",
            value=True,
            help="Enable data quality analysis and annotations"
        )

        create_visualizations = st.checkbox(
            "Generate Visualizations",
            value=True,
            help="Create interactive plots and charts"
        )

        # Model settings
        st.subheader("🤖 Model Settings")
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", 0, 100, 42)

        # Reset button
        if st.button("🔄 Reset Model", help="Clear cached model and retrain"):
            st.session_state.model_trained = False
            st.session_state.model = None
            st.session_state.label_encoder = None
            st.session_state.scaler = None
            st.session_state.imputer = None
            st.session_state.feature_names = None
            st.session_state.training_data = None
            st.session_state.model_accuracy = None
            st.session_state.classification_report_data = None
            st.success("✅ Model reset! Click 'Run Pipeline' to retrain.")

    # Main content
    if st.button("🚀 Run Lithology ML Pipeline", type="primary") or st.session_state.model_trained:

        # Load data (only if not already loaded)
        if st.session_state.training_data is None:
            with st.spinner("📊 Loading data..."):
                try:
                    if uploaded_file is not None:
                        # Load uploaded file
                        data = pd.read_csv(uploaded_file)
                        st.success(f"✅ Uploaded file loaded: {len(data):,} rows")
                    else:
                        # Load sample data
                        data = load_sample_data()
                        st.success(f"✅ Sample data loaded: {len(data):,} rows")

                    # Store in session state
                    st.session_state.training_data = data

                except Exception as e:
                    st.error(f"❌ Failed to load data: {str(e)}")
                    return
        else:
            data = st.session_state.training_data
            st.success(f"✅ Data loaded: {len(data):,} rows")

        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(data):,}")
        with col2:
            st.metric("Total Columns", len(data.columns))
        with col3:
            lithology_count = data['FORCE_2020_LITHOFACIES_LITHOLOGY'].nunique()
            st.metric("Lithology Types", lithology_count)

        # Data quality analysis
        if include_quality_tracking:
            with st.spinner("🔍 Analyzing data quality..."):
                quality_summary = analyze_data_quality(data)

                st.subheader("📋 Data Quality Summary")

                # Missing values
                if quality_summary['missing_values']:
                    st.write("**Missing Values:**")
                    missing_df = pd.DataFrame(list(quality_summary['missing_values'].items()),
                                            columns=['Column', 'Missing Count'])
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("✅ No missing values found")

                # Lithology distribution
                st.write("**Lithology Distribution:**")
                lith_counts = data['FORCE_2020_LITHOFACIES_LITHOLOGY'].value_counts()
                st.bar_chart(lith_counts)

        # Preprocessing (only if not already done)
        if not st.session_state.model_trained:
            with st.spinner("🔧 Preprocessing data..."):
                X, y, label_encoder, imputer, scaler = preprocess_data(data)

                if X is None:
                    st.error("❌ Preprocessing failed")
                    return

                # Store in session state
                st.session_state.label_encoder = label_encoder
                st.session_state.imputer = imputer
                st.session_state.scaler = scaler
                st.session_state.feature_names = X.columns.tolist()

                st.success(f"✅ Data preprocessed: {X.shape[0]} samples, {X.shape[1]} features")
        else:
            # Use cached preprocessing results
            X, y, _, _, _ = preprocess_data(data)
            label_encoder = st.session_state.label_encoder
            st.success(f"✅ Data preprocessed: {X.shape[0]} samples, {X.shape[1]} features")

        # Show lithology types
        st.write("**Lithology Types in Dataset:**")
        lithology_names = list(label_encoder.classes_)
        st.write(f"Found {len(lithology_names)} lithology types:")

        cols = st.columns(4)
        for i, lith in enumerate(lithology_names):
            with cols[i % 4]:
                count = (data['FORCE_2020_LITHOFACIES_LITHOLOGY'] == lith).sum()
                st.write(f"• **{lith}**: {count:,}")

        # Model training (only if not already done)
        if not st.session_state.model_trained:
            with st.spinner("🤖 Training optimized machine learning model..."):
                # Try both Random Forest and Extra Trees, keep the better one
                st.write("Testing Random Forest...")
                rf_model, X_test, y_test, y_pred_rf, y_pred_proba_rf, rf_accuracy = train_model(X, y, 'random_forest')

                st.write("Testing Extra Trees...")
                et_model, _, _, y_pred_et, y_pred_proba_et, et_accuracy = train_model(X, y, 'extra_trees')

                # Choose the better model
                if rf_accuracy >= et_accuracy:
                    model = rf_model
                    y_pred = y_pred_rf
                    y_pred_proba = y_pred_proba_rf
                    accuracy = rf_accuracy
                    model_name = "Random Forest"
                else:
                    model = et_model
                    y_pred = y_pred_et
                    y_pred_proba = y_pred_proba_et
                    accuracy = et_accuracy
                    model_name = "Extra Trees"

                # Store in session state
                st.session_state.model = model
                st.session_state.model_accuracy = accuracy
                st.session_state.model_trained = True

                st.success(f"✅ {model_name} model trained successfully!")
                st.metric("Model Accuracy", f"{accuracy:.3f}")
        else:
            # Use cached model
            model = st.session_state.model
            accuracy = st.session_state.model_accuracy
            # Re-evaluate for display
            X, y, _, _, _ = preprocess_data(data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            st.success(f"✅ Model loaded from cache!")
            st.metric("Model Accuracy", f"{accuracy:.3f}")

        # Model evaluation
        st.subheader("📊 Model Performance")

        # Classification report (cache if not already done)
        if st.session_state.classification_report_data is None:
            class_names = label_encoder.classes_
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.session_state.classification_report_data = report_df
        else:
            report_df = st.session_state.classification_report_data

        # Display classification report (fix data types for display)
        try:
            display_df = report_df.round(3).copy()
            # Convert numeric columns properly
            numeric_cols = ['precision', 'recall', 'f1-score', 'support']
            for col in numeric_cols:
                if col in display_df.columns:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

            # Remove problematic rows for display
            display_df = display_df.dropna()
            st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying classification report: {str(e)}")
            # Fallback: show basic metrics
            st.write(f"**Model Accuracy**: {accuracy:.3f}")
            st.write(f"**Number of Classes**: {len(class_names)}")
            st.write(f"**Classes**: {', '.join(class_names)}")

        # Additional performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            precision_avg = report_df.loc['weighted avg', 'precision']
            st.metric("Weighted Precision", f"{precision_avg:.3f}")
        with col2:
            recall_avg = report_df.loc['weighted avg', 'recall']
            st.metric("Weighted Recall", f"{recall_avg:.3f}")
        with col3:
            f1_avg = report_df.loc['weighted avg', 'f1-score']
            st.metric("Weighted F1-Score", f"{f1_avg:.3f}")

        # Comprehensive Visualizations
        if create_visualizations and st.session_state.model_trained:
            st.subheader("📊 Comprehensive Model Analysis & Visualizations")

            # Get class names from label encoder
            class_names = st.session_state.label_encoder.classes_

            # Ensure we have all required variables for visualizations
            if 'y_test' not in locals() or 'y_pred' not in locals():
                st.warning("⚠️ Visualization data not available. Please run the pipeline again.")
                return

            # Define available parameters for visualizations
            log_params = ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF']
            available_params = [param for param in log_params if param in data.columns]

            # Create tabs for different visualization categories
            viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
                "🎯 Model Performance",
                "📊 Data Analysis",
                "🔍 Feature Analysis",
                "🌈 Lithology Insights",
                "📈 Prediction Analysis"
            ])

            with viz_tab1:
                st.subheader("🎯 Model Performance Analysis")

                # Enhanced Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)

                # Create normalized confusion matrix
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                col1, col2 = st.columns(2)

                with col1:
                    # Raw confusion matrix
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=class_names,
                        y=class_names,
                        color_continuous_scale='Blues',
                        title='Confusion Matrix (Raw Counts)',
                        text_auto=True
                    )
                    fig_cm.update_layout(height=500)
                    st.plotly_chart(fig_cm, use_container_width=True)

                with col2:
                    # Normalized confusion matrix
                    fig_cm_norm = px.imshow(
                        cm_normalized,
                        labels=dict(x="Predicted", y="Actual", color="Accuracy"),
                        x=class_names,
                        y=class_names,
                        color_continuous_scale='RdYlBu_r',
                        title='Confusion Matrix (Normalized)',
                        text_auto='.2f'
                    )
                    fig_cm_norm.update_layout(height=500)
                    st.plotly_chart(fig_cm_norm, use_container_width=True)

                # Per-class performance metrics
                precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)

                metrics_df = pd.DataFrame({
                    'Lithology': class_names,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Support': support
                })

                # Performance metrics visualization
                fig_metrics = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Precision by Lithology', 'Recall by Lithology',
                                  'F1-Score by Lithology', 'Sample Support'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )

                # Precision
                fig_metrics.add_trace(
                    go.Bar(x=metrics_df['Lithology'], y=metrics_df['Precision'],
                           name='Precision', marker_color='lightblue'),
                    row=1, col=1
                )

                # Recall
                fig_metrics.add_trace(
                    go.Bar(x=metrics_df['Lithology'], y=metrics_df['Recall'],
                           name='Recall', marker_color='lightgreen'),
                    row=1, col=2
                )

                # F1-Score
                fig_metrics.add_trace(
                    go.Bar(x=metrics_df['Lithology'], y=metrics_df['F1-Score'],
                           name='F1-Score', marker_color='lightcoral'),
                    row=2, col=1
                )

                # Support
                fig_metrics.add_trace(
                    go.Bar(x=metrics_df['Lithology'], y=metrics_df['Support'],
                           name='Support', marker_color='lightyellow'),
                    row=2, col=2
                )

                fig_metrics.update_layout(height=600, showlegend=False,
                                        title_text="Detailed Performance Metrics by Lithology")
                fig_metrics.update_xaxes(tickangle=45)
                st.plotly_chart(fig_metrics, use_container_width=True)

                # Performance summary table
                st.write("**📋 Detailed Performance Metrics:**")
                st.dataframe(metrics_df.round(3), use_container_width=True)

            with viz_tab2:
                st.subheader("📊 Dataset Analysis & Distribution")

                # Lithology distribution
                lithology_counts = data['FORCE_2020_LITHOFACIES_LITHOLOGY'].value_counts()

                col1, col2 = st.columns(2)

                with col1:
                    # Pie chart of lithology distribution
                    fig_pie = px.pie(
                        values=lithology_counts.values,
                        names=lithology_counts.index,
                        title='Lithology Distribution in Dataset',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Bar chart of lithology counts
                    fig_bar = px.bar(
                        x=lithology_counts.index,
                        y=lithology_counts.values,
                        title='Sample Count by Lithology',
                        labels={'x': 'Lithology', 'y': 'Sample Count'},
                        color=lithology_counts.values,
                        color_continuous_scale='viridis'
                    )
                    fig_bar.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Well log parameter distributions
                st.write("**📈 Well Log Parameter Distributions by Lithology:**")

                # Create parameter distribution plots
                log_params = ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF']
                available_params = [param for param in log_params if param in data.columns]

                if len(available_params) >= 4:
                    # Create 2x2 subplot for first 4 parameters
                    fig_dist = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=available_params[:4]
                    )

                    colors = px.colors.qualitative.Set1

                    for i, param in enumerate(available_params[:4]):
                        row = (i // 2) + 1
                        col = (i % 2) + 1

                        for j, lithology in enumerate(lithology_counts.index[:8]):  # Top 8 lithologies
                            lith_data = data[data['FORCE_2020_LITHOFACIES_LITHOLOGY'] == lithology][param].dropna()

                            fig_dist.add_trace(
                                go.Histogram(
                                    x=lith_data,
                                    name=lithology,
                                    opacity=0.7,
                                    nbinsx=30,
                                    legendgroup=lithology,
                                    showlegend=(i == 0),  # Only show legend for first subplot
                                    marker_color=colors[j % len(colors)]
                                ),
                                row=row, col=col
                            )

                    fig_dist.update_layout(
                        height=600,
                        title_text="Well Log Parameter Distributions by Lithology",
                        barmode='overlay'
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                # Data quality overview
                st.write("**🔍 Data Quality Overview:**")

                quality_metrics = {}
                for param in available_params:
                    total_count = len(data)
                    null_count = data[param].isnull().sum()
                    valid_count = total_count - null_count
                    completeness = (valid_count / total_count) * 100

                    quality_metrics[param] = {
                        'Total Samples': total_count,
                        'Valid Samples': valid_count,
                        'Missing Samples': null_count,
                        'Completeness (%)': completeness,
                        'Min Value': data[param].min(),
                        'Max Value': data[param].max(),
                        'Mean Value': data[param].mean(),
                        'Std Dev': data[param].std()
                    }

                quality_df = pd.DataFrame(quality_metrics).T
                st.dataframe(quality_df.round(2), use_container_width=True)

            with viz_tab3:
                st.subheader("🔍 Feature Analysis & Importance")

                # Feature Importance
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        # Horizontal bar chart
                        fig_importance = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Feature Importance Ranking",
                            color='Importance',
                            color_continuous_scale='viridis',
                            text='Importance'
                        )
                        fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                        fig_importance.update_layout(height=400)
                        st.plotly_chart(fig_importance, use_container_width=True)

                    with col2:
                        # Pie chart of feature importance
                        fig_pie_importance = px.pie(
                            importance_df,
                            values='Importance',
                            names='Feature',
                            title='Feature Importance Distribution',
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_pie_importance.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie_importance, use_container_width=True)

                # Feature correlation analysis
                st.write("**🔗 Feature Correlation Analysis:**")

                # Calculate correlation matrix
                feature_data = data[available_params].select_dtypes(include=[np.number])
                if len(feature_data.columns) > 1:
                    corr_matrix = feature_data.corr()

                    # Create correlation heatmap
                    fig_corr = px.imshow(
                        corr_matrix,
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu',
                        aspect="auto",
                        title="Feature Correlation Matrix",
                        text_auto='.2f'
                    )
                    fig_corr.update_layout(height=500)
                    st.plotly_chart(fig_corr, use_container_width=True)

                # Feature statistics by lithology
                st.write("**📊 Feature Statistics by Lithology:**")

                # Create box plots for each feature
                if len(available_params) >= 2:
                    selected_features = st.multiselect(
                        "Select features to analyze:",
                        available_params,
                        default=available_params[:3]
                    )

                    if selected_features:
                        for feature in selected_features:
                            fig_box = px.box(
                                data,
                                x='FORCE_2020_LITHOFACIES_LITHOLOGY',
                                y=feature,
                                title=f'{feature} Distribution by Lithology',
                                color='FORCE_2020_LITHOFACIES_LITHOLOGY',
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig_box.update_xaxes(tickangle=45)
                            fig_box.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig_box, use_container_width=True)

            with viz_tab4:
                st.subheader("🌈 Lithology Insights & Crossplots")

                # Classic petrophysical crossplots
                st.write("**🎯 Classic Petrophysical Crossplots:**")

                crossplot_options = [
                    ("RHOB vs NPHI", "RHOB", "NPHI"),
                    ("GR vs RDEP", "GR", "RDEP"),
                    ("PEF vs RHOB", "PEF", "RHOB"),
                    ("DTC vs NPHI", "DTC", "NPHI")
                ]

                # Filter available crossplots
                available_crossplots = [
                    (name, x, y) for name, x, y in crossplot_options
                    if x in data.columns and y in data.columns
                ]

                if len(available_crossplots) >= 2:
                    # Create 2x2 subplot for crossplots
                    fig_crossplots = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=[cp[0] for cp in available_crossplots[:4]]
                    )

                    colors = px.colors.qualitative.Set1
                    lithology_colors = {lith: colors[i % len(colors)]
                                      for i, lith in enumerate(lithology_counts.index)}

                    for i, (name, x_param, y_param) in enumerate(available_crossplots[:4]):
                        row = (i // 2) + 1
                        col = (i % 2) + 1

                        for j, lithology in enumerate(lithology_counts.index[:8]):
                            lith_data = data[data['FORCE_2020_LITHOFACIES_LITHOLOGY'] == lithology]

                            fig_crossplots.add_trace(
                                go.Scatter(
                                    x=lith_data[x_param],
                                    y=lith_data[y_param],
                                    mode='markers',
                                    name=lithology,
                                    marker=dict(
                                        color=lithology_colors[lithology],
                                        size=4,
                                        opacity=0.6
                                    ),
                                    legendgroup=lithology,
                                    showlegend=(i == 0)
                                ),
                                row=row, col=col
                            )

                    fig_crossplots.update_layout(
                        height=700,
                        title_text="Petrophysical Crossplots by Lithology"
                    )
                    st.plotly_chart(fig_crossplots, use_container_width=True)

                # 3D scatter plot
                if len(available_params) >= 3:
                    st.write("**🎲 3D Lithology Visualization:**")

                    # Let user select 3 parameters for 3D plot
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_3d = st.selectbox("X-axis:", available_params, index=0)
                    with col2:
                        y_3d = st.selectbox("Y-axis:", available_params, index=1)
                    with col3:
                        z_3d = st.selectbox("Z-axis:", available_params, index=2)

                    if x_3d != y_3d and y_3d != z_3d and x_3d != z_3d:
                        fig_3d = px.scatter_3d(
                            data.sample(n=min(2000, len(data))),  # Sample for performance
                            x=x_3d, y=y_3d, z=z_3d,
                            color='FORCE_2020_LITHOFACIES_LITHOLOGY',
                            title=f'3D Visualization: {x_3d} vs {y_3d} vs {z_3d}',
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            opacity=0.7
                        )
                        fig_3d.update_layout(height=600)
                        st.plotly_chart(fig_3d, use_container_width=True)

                # Lithology characteristics table
                st.write("**📋 Lithology Characteristics Summary:**")

                lith_summary = []
                for lithology in lithology_counts.index:
                    lith_data = data[data['FORCE_2020_LITHOFACIES_LITHOLOGY'] == lithology]

                    summary_row = {'Lithology': lithology, 'Sample_Count': len(lith_data)}

                    for param in available_params:
                        if param in lith_data.columns:
                            summary_row[f'{param}_Mean'] = lith_data[param].mean()
                            summary_row[f'{param}_Std'] = lith_data[param].std()

                    lith_summary.append(summary_row)

                lith_summary_df = pd.DataFrame(lith_summary)
                st.dataframe(lith_summary_df.round(2), use_container_width=True)

            with viz_tab5:
                st.subheader("📈 Prediction Analysis & Model Insights")

                # Check if predictions are available from session state or local variables
                pred_names = None
                pred_confidence = None

                # Try to get predictions from the current scope or recreate them
                try:
                    if 'prediction_names' in locals() and prediction_names is not None:
                        pred_names = prediction_names
                        pred_confidence = prediction_confidence
                    elif st.session_state.model is not None:
                        # Recreate predictions for visualization
                        X_viz, _, _, _, _ = preprocess_data(data)
                        if X_viz is not None:
                            pred_full = st.session_state.model.predict(X_viz)
                            pred_names = st.session_state.label_encoder.inverse_transform(pred_full)
                            pred_proba = st.session_state.model.predict_proba(X_viz)
                            pred_confidence = np.max(pred_proba, axis=1)
                except Exception as e:
                    st.error(f"Error getting predictions for visualization: {str(e)}")

                if pred_names is not None and pred_confidence is not None:
                    # Prediction confidence analysis
                    st.write("**🎯 Prediction Confidence Analysis:**")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Confidence distribution histogram
                        fig_conf_hist = px.histogram(
                            x=pred_confidence,
                            nbins=30,
                            title='Prediction Confidence Distribution',
                            labels={'x': 'Confidence Score', 'y': 'Count'},
                            color_discrete_sequence=['skyblue']
                        )
                        fig_conf_hist.add_vline(x=np.mean(pred_confidence),
                                              line_dash="dash", line_color="red",
                                              annotation_text=f"Mean: {np.mean(pred_confidence):.3f}")
                        st.plotly_chart(fig_conf_hist, use_container_width=True)

                    with col2:
                        # Confidence by predicted lithology
                        pred_conf_df = pd.DataFrame({
                            'Predicted_Lithology': pred_names,
                            'Confidence': pred_confidence
                        })

                        fig_conf_box = px.box(
                            pred_conf_df,
                            x='Predicted_Lithology',
                            y='Confidence',
                            title='Confidence by Predicted Lithology',
                            color='Predicted_Lithology',
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig_conf_box.update_xaxes(tickangle=45)
                        fig_conf_box.update_layout(showlegend=False)
                        st.plotly_chart(fig_conf_box, use_container_width=True)

                    # High vs Low confidence predictions
                    st.write("**🔍 High vs Low Confidence Predictions:**")

                    high_conf_threshold = 0.8
                    low_conf_threshold = 0.6

                    high_conf_count = (pred_confidence >= high_conf_threshold).sum()
                    medium_conf_count = ((pred_confidence >= low_conf_threshold) &
                                       (pred_confidence < high_conf_threshold)).sum()
                    low_conf_count = (pred_confidence < low_conf_threshold).sum()

                    conf_categories = pd.DataFrame({
                        'Confidence_Level': ['High (≥0.8)', 'Medium (0.6-0.8)', 'Low (<0.6)'],
                        'Count': [high_conf_count, medium_conf_count, low_conf_count],
                        'Percentage': [
                            high_conf_count/len(pred_confidence)*100,
                            medium_conf_count/len(pred_confidence)*100,
                            low_conf_count/len(pred_confidence)*100
                        ]
                    })

                    fig_conf_cat = px.bar(
                        conf_categories,
                        x='Confidence_Level',
                        y='Count',
                        title='Prediction Confidence Categories',
                        color='Percentage',
                        color_continuous_scale='RdYlGn',
                        text='Count'
                    )
                    fig_conf_cat.update_traces(textposition='outside')
                    st.plotly_chart(fig_conf_cat, use_container_width=True)

                    # Model uncertainty analysis
                    st.write("**⚠️ Model Uncertainty Analysis:**")

                    # Find most uncertain predictions (lowest confidence)
                    uncertain_indices = np.argsort(pred_confidence)[:20]  # Top 20 most uncertain

                    uncertain_data = data.iloc[uncertain_indices].copy()
                    uncertain_data['Predicted_Lithology'] = [pred_names[i] for i in uncertain_indices]
                    uncertain_data['Confidence'] = [pred_confidence[i] for i in uncertain_indices]

                    st.write("**🔍 Most Uncertain Predictions (Lowest 20 Confidence Scores):**")
                    display_cols = ['FORCE_2020_LITHOFACIES_LITHOLOGY', 'Predicted_Lithology', 'Confidence'] + available_params[:3]
                    available_display_cols = [col for col in display_cols if col in uncertain_data.columns]
                    st.dataframe(uncertain_data[available_display_cols].round(3), use_container_width=True)

                else:
                    st.warning("⚠️ No prediction data available for analysis. Please run the pipeline first.")

        # Make predictions on full dataset
        if st.session_state.model is not None:
            # Initialize variables to avoid errors
            predictions_full = None
            prediction_names = None
            prediction_confidence = None
            quality_annotations = pd.DataFrame()

            try:
                with st.spinner("🔮 Making predictions on full dataset..."):
                    # Preprocess full dataset
                    X_full, _, _, _, _ = preprocess_data(data)
                    if X_full is not None:
                        predictions_full = st.session_state.model.predict(X_full)
                        prediction_names = st.session_state.label_encoder.inverse_transform(predictions_full)

                        # Get prediction probabilities for confidence
                        prediction_proba = st.session_state.model.predict_proba(X_full)
                        prediction_confidence = np.max(prediction_proba, axis=1)

                        # Create quality annotations if enabled
                        if include_quality_tracking:
                            quality_annotations = create_quality_annotations(data, X_full, st.session_state.imputer, st.session_state.scaler)
                        else:
                            quality_annotations = pd.DataFrame(index=data.index)

            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                # Set default values to prevent download errors
                prediction_names = ['Unknown'] * len(data)
                prediction_confidence = [0.0] * len(data)
                quality_annotations = pd.DataFrame(index=data.index)

        # Download Section
        st.subheader("💾 Download Results")

        # Only show download if predictions are available
        if prediction_names is not None and len(prediction_names) > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**📊 Standard Results**")

                try:
                    # Standard results with predictions
                    standard_results = data.copy()
                    if len(prediction_names) == len(standard_results):
                        standard_results['ML_Predicted_Lithology'] = prediction_names
                        if prediction_confidence is not None:
                            standard_results['Prediction_Confidence'] = prediction_confidence

                    # Convert to CSV
                    csv_buffer = io.StringIO()
                    standard_results.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()

                    st.download_button(
                        label="📥 Download Standard Results",
                        data=csv_data,
                        file_name=f"lithology_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download dataset with ML predictions and confidence scores"
                    )

                    st.info(f"📊 {len(standard_results):,} rows, {len(standard_results.columns)} columns")

                    # Show prediction summary
                    if prediction_names is not None:
                        st.write("**Prediction Summary:**")
                        pred_summary = pd.Series(prediction_names).value_counts()
                        st.bar_chart(pred_summary)

                except Exception as e:
                    st.error(f"Error creating standard results: {str(e)}")

            with col2:
                st.write("**🔬 Results with Quality Tracking**")

                try:
                    # Enhanced dataset with quality annotations
                    enhanced_results = data.copy()
                    if len(prediction_names) == len(enhanced_results):
                        enhanced_results['ML_Predicted_Lithology'] = prediction_names
                        if prediction_confidence is not None:
                            enhanced_results['Prediction_Confidence'] = prediction_confidence

                            # Add confidence categories
                            enhanced_results['Confidence_Category'] = pd.cut(
                                prediction_confidence,
                                bins=[0, 0.6, 0.8, 1.0],
                                labels=['Low', 'Medium', 'High']
                            )

                    # Add quality annotations
                    if include_quality_tracking and not quality_annotations.empty:
                        for col in quality_annotations.columns:
                            enhanced_results[f'Quality_{col}'] = quality_annotations[col]

                    # Convert to CSV
                    enhanced_csv_buffer = io.StringIO()
                    enhanced_results.to_csv(enhanced_csv_buffer, index=False)
                    enhanced_csv_data = enhanced_csv_buffer.getvalue()

                    st.download_button(
                        label="📥 Download with Quality Tracking",
                        data=enhanced_csv_data,
                        file_name=f"lithology_with_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download dataset with quality annotations, ML predictions, and confidence scores"
                    )

                    st.success(f"🔬 {len(enhanced_results):,} rows, {len(enhanced_results.columns)} columns")

                    # Show confidence distribution
                    if prediction_confidence is not None:
                        st.write("**Prediction Confidence Distribution:**")
                        if 'Confidence_Category' in enhanced_results.columns:
                            conf_dist = enhanced_results['Confidence_Category'].value_counts()
                            st.bar_chart(conf_dist)

                except Exception as e:
                    st.error(f"Error creating enhanced results: {str(e)}")

            with col3:
                st.write("**🔬 Processed Dataset for Visualizations**")

                try:
                    # Create processed dataset with detailed annotations
                    if st.session_state.model_trained:
                        # Get processed features used for training
                        X_processed, _, _, _, _ = preprocess_data(data)

                        # Create comprehensive processed dataset
                        processed_dataset = create_processed_dataset_with_annotations(
                            data,
                            X_processed,
                            quality_annotations,
                            prediction_names,
                            prediction_confidence
                        )

                        # Convert to CSV
                        processed_csv_buffer = io.StringIO()
                        processed_dataset.to_csv(processed_csv_buffer, index=False)
                        processed_csv_data = processed_csv_buffer.getvalue()

                        st.download_button(
                            label="📥 Download Processed Dataset",
                            data=processed_csv_data,
                            file_name=f"processed_dataset_for_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download the processed dataset used for visualizations with outlier detection, null value treatment, and processing details"
                        )

                        st.info(f"🔬 {len(processed_dataset):,} rows, {len(processed_dataset.columns)} columns")

                        # Show processing summary
                        st.write("**Processing Summary:**")
                        null_treated = (processed_dataset.get('Quality_Has_Null_Values', pd.Series([False])) == True).sum()
                        outliers_detected = sum([
                            (processed_dataset.get(f'{col}_Is_Outlier', pd.Series([False])) == True).sum()
                            for col in ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF']
                            if f'{col}_Is_Outlier' in processed_dataset.columns
                        ])

                        st.write(f"• **Null values treated**: {null_treated:,} rows")
                        st.write(f"• **Outliers detected**: {outliers_detected:,} values")
                        st.write(f"• **Processing method**: Median imputation + StandardScaler")
                        st.write(f"• **Outlier treatment**: Preserved for geological validity")

                    else:
                        st.warning("⚠️ Model not trained yet. Please run the pipeline first.")

                except Exception as e:
                    st.error(f"Error creating processed dataset: {str(e)}")
        else:
            st.warning("⚠️ No predictions available. Please run the pipeline first.")

        # Dataset preview
        if include_quality_tracking:
            with st.expander("🔬 Quality Tracking Preview", expanded=False):
                st.write("**Quality tracking includes:**")
                st.write("• Null value detection and flags")
                st.write("• Data quality scores")
                st.write("• Processing techniques applied")
                st.write("• ML predictions with lithology names")

                if not quality_annotations.empty:
                    st.write("**Sample of quality columns:**")
                    st.dataframe(quality_annotations.head(), use_container_width=True)

        # Processed Dataset Preview
        if st.session_state.model_trained:
            with st.expander("🔬 Processed Dataset Preview (Used for Visualizations)", expanded=False):
                st.write("**The processed dataset includes:**")
                st.write("• **Original values**: Raw well log measurements")
                st.write("• **Processed values**: Scaled values used for ML training (suffix: _Processed)")
                st.write("• **Outlier detection**: Flags and bounds for each parameter (suffix: _Is_Outlier, _Outlier_Bounds)")
                st.write("• **Quality annotations**: Null value flags, quality scores, processing techniques")
                st.write("• **ML predictions**: Predicted lithology with confidence scores")
                st.write("• **Processing metadata**: Timestamps, methods used, treatment details")

                try:
                    # Create sample processed dataset for preview
                    X_processed, _, _, _, _ = preprocess_data(data)
                    sample_processed = create_processed_dataset_with_annotations(
                        data.head(10),
                        X_processed.head(10),
                        quality_annotations.head(10) if not quality_annotations.empty else pd.DataFrame(),
                        prediction_names[:10] if prediction_names is not None else None,
                        prediction_confidence[:10] if prediction_confidence is not None else None
                    )

                    st.write("**Sample of processed dataset (first 10 rows):**")

                    # Show different column categories
                    col_tabs = st.tabs(["Original Data", "Processed Values", "Outlier Detection", "Quality Info", "Predictions"])

                    with col_tabs[0]:
                        original_cols = [col for col in ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF', 'FORCE_2020_LITHOFACIES_LITHOLOGY'] if col in sample_processed.columns]
                        if original_cols:
                            st.dataframe(sample_processed[original_cols], use_container_width=True)

                    with col_tabs[1]:
                        processed_cols = [col for col in sample_processed.columns if col.endswith('_Processed')]
                        if processed_cols:
                            st.dataframe(sample_processed[processed_cols], use_container_width=True)
                        else:
                            st.write("No processed columns available")

                    with col_tabs[2]:
                        outlier_cols = [col for col in sample_processed.columns if '_Is_Outlier' in col or '_Outlier_Bounds' in col]
                        if outlier_cols:
                            st.dataframe(sample_processed[outlier_cols], use_container_width=True)
                        else:
                            st.write("No outlier detection columns available")

                    with col_tabs[3]:
                        quality_cols = [col for col in sample_processed.columns if col.startswith('Quality_') or 'Treatment' in col or 'Processing' in col]
                        if quality_cols:
                            st.dataframe(sample_processed[quality_cols], use_container_width=True)
                        else:
                            st.write("No quality information columns available")

                    with col_tabs[4]:
                        prediction_cols = [col for col in sample_processed.columns if 'Predicted' in col or 'Confidence' in col]
                        if prediction_cols:
                            st.dataframe(sample_processed[prediction_cols], use_container_width=True)
                        else:
                            st.write("No prediction columns available")

                    st.write(f"**Total columns in processed dataset**: {len(sample_processed.columns)}")
                    st.write(f"**Column categories**:")
                    st.write(f"• Original data: {len(original_cols)} columns")
                    st.write(f"• Processed values: {len(processed_cols)} columns")
                    st.write(f"• Outlier detection: {len(outlier_cols)} columns")
                    st.write(f"• Quality information: {len(quality_cols)} columns")
                    st.write(f"• Predictions: {len(prediction_cols)} columns")

                except Exception as e:
                    st.error(f"Error creating processed dataset preview: {str(e)}")

    else:
        # Instructions when not running
        st.markdown("""
        ## 🎯 Welcome to the Lithology ML Pipeline!

        This application provides:

        ### 🔬 **Key Features**
        - **Comprehensive Data Quality Analysis**: Automatic detection of null values and data issues
        - **Multiple Lithology Types**: Supports 10+ lithology types including Sandstone, Shale, Limestone, Dolomite, etc.
        - **Machine Learning Model**: Random Forest classifier with optimized parameters
        - **Quality Tracking**: Detailed annotations showing data quality and preprocessing steps

        ### 📊 **What You'll Get**
        - **Standard Results**: Dataset with ML predictions showing actual lithology names
        - **Quality Tracking**: Complete data quality annotations and preprocessing details
        - **Interactive Visualizations**: Confusion matrix and feature importance analysis
        - **Performance Metrics**: Detailed classification report with precision, recall, and F1-scores

        ### 🚀 **Getting Started**
        1. Configure your settings in the sidebar
        2. Optionally upload your own CSV data (or use sample data with 5000+ samples)
        3. Click "Run Lithology ML Pipeline"
        4. Download your results with quality tracking

        ### 📋 **Supported Lithology Types**
        - Sandstone, Shale, Limestone, Dolomite
        - Anhydrite, Salt, Coal, Marl
        - Sandstone_Shaly, Limestone_Shaly

        **Ready to classify your lithology data?** 🪨📊
        """)

if __name__ == "__main__":
    main()
