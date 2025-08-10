"""
🪨 LITHOLOGY CLASSIFICATION ML PIPELINE
=====================================
A comprehensive machine learning pipeline for predicting lithology classes from well log data.

Features:
- Data preprocessing with intelligent missing value handling
- Random Forest and XGBoost classifiers
- Comprehensive evaluation metrics and visualizations
- Interactive Plotly plots for geoscientist interpretation
- Model persistence and inference capabilities
- Professional presentation-ready outputs

Author: ONGC Petrophysical Analysis Team
Date: 2025-01-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                           accuracy_score, precision_recall_fscore_support)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib

# System imports
import os
import glob
from datetime import datetime
import json

class LithologyMLPipeline:
    """
    Complete ML Pipeline for Lithology Classification
    """

    def __init__(self, data_file="litho_data/xeek_subset_example.csv", results_dir="model_results",
                 max_samples=None, memory_efficient=False):
        """
        Initialize the ML pipeline with memory optimization and quality tracking

        Args:
            data_file (str): Path to the CSV file with well log data
            results_dir (str): Directory to save models and results
            max_samples (int): Maximum number of samples to use (None for all)
            memory_efficient (bool): Enable memory optimization techniques
        """
        self.data_file = data_file
        self.results_dir = "model_results"
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF']
        self.target_column = 'FORCE_2020_LITHOFACIES_LITHOLOGY'

        # Force full data usage (no memory optimization)
        self.max_samples = None
        self.memory_efficient = False
        self.sample_used = False

        # Enhanced data quality tracking
        self.quality_annotations = pd.DataFrame()
        self.preprocessing_log = []
        self.technique_performance = {}
        self.best_techniques = {}

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/visualizations", exist_ok=True)

        print("🚀 Lithology ML Pipeline Initialized")
        print(f"📁 Data File: {self.data_file}")
        print(f"💾 Results Directory: {self.results_dir}")
        print("🧠 Memory optimization: DISABLED (using full dataset)")
        print("📋 Enhanced data quality tracking: ENABLED")

    def load_and_combine_data(self):
        """
        Load the specified CSV file as the dataset

        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("\n📊 LOADING DATA FROM FILE")
        print("=" * 50)

        if not os.path.isfile(self.data_file):
            raise FileNotFoundError(f"CSV file not found: {self.data_file}")

        try:
            df = pd.read_csv(self.data_file)
            print(f"   ✅ {os.path.basename(self.data_file)}: {len(df):,} rows, {len(df.columns)} columns")
            df['SOURCE_FILE'] = os.path.basename(self.data_file)
        except Exception as e:
            print(f"   ❌ Error loading {self.data_file}: {str(e)}")
            raise

        print(f"\n📈 DATASET SUMMARY:")
        print(f"   📊 Total rows: {len(df):,}")
        print(f"   📋 Total columns: {len(df.columns)}")
        print(f"   🏢 Wells: {df['WELL'].nunique() if 'WELL' in df.columns else 'N/A'}")

        self.raw_data = df
        return df

    def _optimize_memory_usage(self, df):
        """
        Optimize memory usage of the dataframe

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Memory-optimized dataframe
        """
        print(f"\n🧠 MEMORY OPTIMIZATION")
        print("=" * 30)

        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        print(f"📊 Initial memory usage: {initial_memory:.1f} MB")

        # Apply sampling if dataset is too large
        if len(df) > 100000 or initial_memory > 500:  # 500 MB threshold
            if self.max_samples is None:
                # Auto-determine sample size based on memory
                if initial_memory > 1000:  # 1 GB
                    sample_size = 50000
                elif initial_memory > 500:  # 500 MB
                    sample_size = 75000
                else:
                    sample_size = 100000
            else:
                sample_size = min(self.max_samples, len(df))

            if sample_size < len(df):
                print(f"⚠️  Large dataset detected: {len(df):,} samples")
                print(f"🎯 Applying stratified sampling: {sample_size:,} samples")

                # Stratified sampling to maintain lithology distribution
                if self.target_column in df.columns:
                    df_sampled = df.groupby(self.target_column, group_keys=False).apply(
                        lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(df)))),
                                         random_state=42)
                    ).reset_index(drop=True)
                else:
                    df_sampled = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

                df = df_sampled
                self.sample_used = True
                print(f"✅ Sampled dataset: {len(df):,} samples")

        # Optimize data types
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')

        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        print(f"📊 Final memory usage: {final_memory:.1f} MB")
        print(f"💾 Memory saved: {initial_memory - final_memory:.1f} MB ({((initial_memory - final_memory) / initial_memory * 100):.1f}%)")

        return df

    def _initialize_quality_tracking(self, data):
        """Initialize comprehensive data quality tracking system"""
        print(f"\n📋 INITIALIZING QUALITY TRACKING SYSTEM")
        print("=" * 50)

        # Initialize quality annotations DataFrame
        self.quality_annotations = pd.DataFrame(index=data.index)
        self.quality_annotations['Row_ID'] = range(len(data))
        self.quality_annotations['Has_Null_Values'] = False
        self.quality_annotations['Has_Outliers'] = False
        self.quality_annotations['Quality_Issues'] = ''
        self.quality_annotations['Techniques_Applied'] = ''
        self.quality_annotations['Best_Technique'] = ''
        self.quality_annotations['Processing_Notes'] = ''

        # Initialize technique performance tracking
        self.technique_performance = {
            'imputation': {},
            'outlier_handling': {},
            'scaling': {}
        }

        print(f"✅ Quality tracking initialized for {len(data):,} rows")
        return self.quality_annotations

    def _detect_and_log_null_values(self, data, features):
        """Detect null values and log quality issues"""
        print(f"\n🔍 DETECTING NULL VALUES")

        for feature in features:
            if feature in data.columns:
                null_mask = data[feature].isnull()
                null_count = null_mask.sum()

                if null_count > 0:
                    print(f"   📊 {feature}: {null_count:,} null values ({null_count/len(data)*100:.1f}%)")

                    # Update quality annotations
                    self.quality_annotations.loc[null_mask, 'Has_Null_Values'] = True

                    # Add to quality issues
                    current_issues = self.quality_annotations.loc[null_mask, 'Quality_Issues']
                    new_issues = current_issues.apply(lambda x: f"{x}; Null in {feature}" if x else f"Null in {feature}")
                    self.quality_annotations.loc[null_mask, 'Quality_Issues'] = new_issues

        total_null_rows = self.quality_annotations['Has_Null_Values'].sum()
        print(f"📈 Total rows with null values: {total_null_rows:,} ({total_null_rows/len(data)*100:.1f}%)")

        return total_null_rows

    def _detect_and_log_outliers(self, data, features, method='iqr', threshold=3.0):
        """Detect outliers using multiple methods and log quality issues"""
        print(f"\n🎯 DETECTING OUTLIERS (Method: {method.upper()})")

        outlier_counts = {}

        for feature in features:
            if feature in data.columns and data[feature].dtype in ['float64', 'int64']:
                if method == 'iqr':
                    Q1 = data[feature].quantile(0.25)
                    Q3 = data[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_mask = (data[feature] < lower_bound) | (data[feature] > upper_bound)

                elif method == 'zscore':
                    z_scores = np.abs((data[feature] - data[feature].mean()) / data[feature].std())
                    outlier_mask = z_scores > threshold

                elif method == 'modified_zscore':
                    median = data[feature].median()
                    mad = np.median(np.abs(data[feature] - median))
                    modified_z_scores = 0.6745 * (data[feature] - median) / mad
                    outlier_mask = np.abs(modified_z_scores) > threshold

                outlier_count = outlier_mask.sum()
                outlier_counts[feature] = outlier_count

                if outlier_count > 0:
                    print(f"   📊 {feature}: {outlier_count:,} outliers ({outlier_count/len(data)*100:.1f}%)")

                    # Update quality annotations
                    self.quality_annotations.loc[outlier_mask, 'Has_Outliers'] = True

                    # Add to quality issues
                    current_issues = self.quality_annotations.loc[outlier_mask, 'Quality_Issues']
                    new_issues = current_issues.apply(lambda x: f"{x}; Outlier in {feature}" if x else f"Outlier in {feature}")
                    self.quality_annotations.loc[outlier_mask, 'Quality_Issues'] = new_issues

        total_outlier_rows = self.quality_annotations['Has_Outliers'].sum()
        print(f"📈 Total rows with outliers: {total_outlier_rows:,} ({total_outlier_rows/len(data)*100:.1f}%)")

        return outlier_counts

    def _test_imputation_techniques(self, data, features):
        """Test multiple imputation techniques and track performance"""
        print(f"\n🧪 TESTING IMPUTATION TECHNIQUES")

        from sklearn.impute import SimpleImputer, KNNImputer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        techniques = {
            'median': SimpleImputer(strategy='median'),
            'mean': SimpleImputer(strategy='mean'),
            'knn': KNNImputer(n_neighbors=5),
            'iterative': IterativeImputer(random_state=42, max_iter=10)
        }

        # Test each technique on a sample of data
        sample_size = min(1000, len(data))
        sample_data = data[features].sample(n=sample_size, random_state=42)

        technique_scores = {}

        for technique_name, imputer in techniques.items():
            try:
                # Create artificial missing values for testing
                test_data = sample_data.copy()
                missing_mask = np.random.random(test_data.shape) < 0.1  # 10% missing
                test_data_missing = test_data.copy()
                test_data_missing[missing_mask] = np.nan

                # Apply imputation
                imputed_data = pd.DataFrame(
                    imputer.fit_transform(test_data_missing),
                    columns=features
                )

                # Calculate reconstruction error (MSE)
                mse = np.mean((test_data.values - imputed_data.values) ** 2)
                technique_scores[technique_name] = mse

                print(f"   📊 {technique_name.capitalize()}: MSE = {mse:.6f}")

            except Exception as e:
                print(f"   ⚠️ {technique_name.capitalize()}: Failed ({str(e)})")
                technique_scores[technique_name] = float('inf')

        # Find best technique
        best_technique = min(technique_scores.keys(), key=lambda x: technique_scores[x])
        self.technique_performance['imputation'] = technique_scores
        self.best_techniques['imputation'] = best_technique

        print(f"🏆 Best imputation technique: {best_technique.upper()} (MSE: {technique_scores[best_technique]:.6f})")

        return best_technique, techniques[best_technique]

    def _normalize_mixed_data_types(self, y):
        """Handle mixed numerical codes and text labels in target column"""
        print(f"🔧 Normalizing mixed data types...")

        # Convert all target values to strings
        y_normalized = y.astype(str)

        # Map numerical codes to lithology names
        numerical_to_lithology = {
            '65000': 'Sandstone',
            '30000': 'Shale',
            '70000': 'Limestone',
            '65030': 'Sandstone_Shaly',
            '80000': 'Dolomite',
            '88000': 'Anhydrite',
            '70032': 'Limestone_Shaly',
            '74000': 'Marl',
            '86000': 'Salt',
            '99000': 'Coal'
        }

        # Apply mapping
        y_mapped = y_normalized.map(numerical_to_lithology).fillna(y_normalized)

        # Show mapping results
        unique_before = set(y_normalized.unique())
        unique_after = set(y_mapped.unique())
        mapped_count = len(unique_before - unique_after)

        if mapped_count > 0:
            print(f"✅ Mapped {mapped_count} numerical codes to lithology names")
            print(f"📊 Before: {len(unique_before)} unique values")
            print(f"📊 After: {len(unique_after)} unique values")

        return y_mapped

    def _detect_and_log_null_values(self, data, features):
        """Detect null values and log quality issues"""
        print(f"\n🔍 DETECTING NULL VALUES")

        for feature in features:
            if feature in data.columns:
                null_mask = data[feature].isnull()
                null_count = null_mask.sum()

                if null_count > 0:
                    print(f"   📊 {feature}: {null_count:,} null values ({null_count/len(data)*100:.1f}%)")

                    # Update quality annotations
                    self.quality_annotations.loc[null_mask, 'Has_Null_Values'] = True

                    # Add to quality issues
                    current_issues = self.quality_annotations.loc[null_mask, 'Quality_Issues']
                    new_issues = current_issues.apply(lambda x: f"{x}; Null in {feature}" if x else f"Null in {feature}")
                    self.quality_annotations.loc[null_mask, 'Quality_Issues'] = new_issues

        total_null_rows = self.quality_annotations['Has_Null_Values'].sum()
        print(f"📈 Total rows with null values: {total_null_rows:,} ({total_null_rows/len(data)*100:.1f}%)")

        return total_null_rows

    def _detect_and_log_outliers(self, data, features, method='iqr', threshold=3.0):
        """Detect outliers using multiple methods and log quality issues"""
        print(f"\n🎯 DETECTING OUTLIERS (Method: {method.upper()})")

        outlier_counts = {}

        for feature in features:
            if feature in data.columns and data[feature].dtype in ['float64', 'int64']:
                if method == 'iqr':
                    Q1 = data[feature].quantile(0.25)
                    Q3 = data[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_mask = (data[feature] < lower_bound) | (data[feature] > upper_bound)

                elif method == 'zscore':
                    z_scores = np.abs((data[feature] - data[feature].mean()) / data[feature].std())
                    outlier_mask = z_scores > threshold

                elif method == 'modified_zscore':
                    median = data[feature].median()
                    mad = np.median(np.abs(data[feature] - median))
                    modified_z_scores = 0.6745 * (data[feature] - median) / mad
                    outlier_mask = np.abs(modified_z_scores) > threshold

                outlier_count = outlier_mask.sum()
                outlier_counts[feature] = outlier_count

                if outlier_count > 0:
                    print(f"   📊 {feature}: {outlier_count:,} outliers ({outlier_count/len(data)*100:.1f}%)")

                    # Update quality annotations
                    self.quality_annotations.loc[outlier_mask, 'Has_Outliers'] = True

                    # Add to quality issues
                    current_issues = self.quality_annotations.loc[outlier_mask, 'Quality_Issues']
                    new_issues = current_issues.apply(lambda x: f"{x}; Outlier in {feature}" if x else f"Outlier in {feature}")
                    self.quality_annotations.loc[outlier_mask, 'Quality_Issues'] = new_issues

        total_outlier_rows = self.quality_annotations['Has_Outliers'].sum()
        print(f"📈 Total rows with outliers: {total_outlier_rows:,} ({total_outlier_rows/len(data)*100:.1f}%)")

        return outlier_counts

    def create_enhanced_dataset_with_annotations(self, original_data, predictions=None):
        """
        Create enhanced dataset with comprehensive quality annotations and predictions

        Args:
            original_data (pd.DataFrame): Original dataset
            predictions (array-like, optional): Model predictions

        Returns:
            pd.DataFrame: Enhanced dataset with quality annotations
        """
        print(f"\n📋 CREATING ENHANCED DATASET WITH QUALITY ANNOTATIONS")
        print("=" * 60)

        # Start with original data
        enhanced_data = original_data.copy()

        # Ensure quality annotations exist and match the data
        if self.quality_annotations.empty:
            print("⚠️ No quality annotations found. Run preprocessing first.")
            return enhanced_data

        # Align indices between original data and quality annotations
        common_indices = enhanced_data.index.intersection(self.quality_annotations.index)
        enhanced_data = enhanced_data.loc[common_indices]
        quality_subset = self.quality_annotations.loc[common_indices]

        # Add quality annotation columns
        enhanced_data['Data_Quality_Has_Null_Values'] = quality_subset['Has_Null_Values']
        enhanced_data['Data_Quality_Has_Outliers'] = quality_subset['Has_Outliers']
        enhanced_data['Data_Quality_Issues'] = quality_subset['Quality_Issues']
        enhanced_data['Preprocessing_Techniques_Applied'] = quality_subset['Techniques_Applied']
        enhanced_data['Best_Technique_Used'] = quality_subset['Best_Technique']
        enhanced_data['Processing_Notes'] = quality_subset['Processing_Notes']

        # Add technique performance summary
        if self.technique_performance:
            # Create technique performance summary
            technique_summary = []

            # Imputation performance
            if 'imputation' in self.technique_performance:
                imp_perf = self.technique_performance['imputation']
                best_imp = min(imp_perf.keys(), key=lambda x: imp_perf[x]) if imp_perf else 'None'
                technique_summary.append(f"Best_Imputation: {best_imp}")

            # Add overall best technique summary
            enhanced_data['Best_Technique_Summary'] = '; '.join(technique_summary) if technique_summary else 'Standard Processing'

        # Add predictions if available
        if predictions is not None:
            if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
                # Convert numerical predictions to lithology names
                if isinstance(predictions[0], (int, np.integer)):
                    prediction_names = self.label_encoder.inverse_transform(predictions)
                else:
                    prediction_names = predictions
                enhanced_data['ML_Predicted_Lithology'] = prediction_names
            else:
                enhanced_data['ML_Predicted_Lithology'] = predictions

        # Add data quality score
        quality_score = []
        for idx in enhanced_data.index:
            score = 100  # Start with perfect score

            if idx in quality_subset.index:
                # Deduct points for quality issues
                if quality_subset.loc[idx, 'Has_Null_Values']:
                    score -= 30
                if quality_subset.loc[idx, 'Has_Outliers']:
                    score -= 20

                # Count number of issues
                issues = quality_subset.loc[idx, 'Quality_Issues']
                if issues:
                    issue_count = len(issues.split(';'))
                    score -= min(issue_count * 10, 50)  # Max 50 points deduction

            quality_score.append(max(0, score))  # Ensure non-negative

        enhanced_data['Data_Quality_Score'] = quality_score

        # Add preprocessing metadata
        enhanced_data['Preprocessing_Timestamp'] = pd.Timestamp.now()
        enhanced_data['Pipeline_Version'] = 'Enhanced_v2.0'

        # Reorder columns for better readability
        original_cols = [col for col in original_data.columns if col in enhanced_data.columns]
        quality_cols = [
            'Data_Quality_Score',
            'Data_Quality_Has_Null_Values',
            'Data_Quality_Has_Outliers',
            'Data_Quality_Issues',
            'Preprocessing_Techniques_Applied',
            'Best_Technique_Used',
            'Best_Technique_Summary',
            'Processing_Notes'
        ]

        prediction_cols = [col for col in enhanced_data.columns if 'Predicted' in col]
        metadata_cols = ['Preprocessing_Timestamp', 'Pipeline_Version']

        # Final column order
        final_columns = original_cols + quality_cols + prediction_cols + metadata_cols
        enhanced_data = enhanced_data[[col for col in final_columns if col in enhanced_data.columns]]

        print(f"✅ Enhanced dataset created with {len(enhanced_data):,} rows and {len(enhanced_data.columns)} columns")
        print(f"📊 Quality annotation columns added: {len(quality_cols)}")

        # Summary statistics
        null_rows = enhanced_data['Data_Quality_Has_Null_Values'].sum()
        outlier_rows = enhanced_data['Data_Quality_Has_Outliers'].sum()
        avg_quality_score = enhanced_data['Data_Quality_Score'].mean()

        print(f"📈 Quality Summary:")
        print(f"   • Rows with null values: {null_rows:,} ({null_rows/len(enhanced_data)*100:.1f}%)")
        print(f"   • Rows with outliers: {outlier_rows:,} ({outlier_rows/len(enhanced_data)*100:.1f}%)")
        print(f"   • Average quality score: {avg_quality_score:.1f}/100")

        return enhanced_data

    def save_enhanced_dataset(self, enhanced_data, filename=None):
        """
        Save enhanced dataset with quality annotations to CSV

        Args:
            enhanced_data (pd.DataFrame): Enhanced dataset
            filename (str, optional): Custom filename

        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"lithology_enhanced_dataset_{timestamp}.csv"

        filepath = os.path.join(self.results_dir, filename)
        enhanced_data.to_csv(filepath, index=False)

        print(f"💾 Enhanced dataset saved: {filepath}")
        print(f"📊 File size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")

        return filepath

    def _check_memory_usage(self):
        """
        Check current memory usage and provide recommendations
        """
        import psutil

        # Get memory info
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        used_percent = memory.percent

        print(f"\n💾 MEMORY STATUS:")
        print(f"   📊 Available RAM: {available_gb:.1f} GB")
        print(f"   📈 Memory usage: {used_percent:.1f}%")

        # Provide recommendations
        if available_gb < 2:
            print(f"   ⚠️  Low memory warning! Consider:")
            print(f"      • Reducing max_samples parameter")
            print(f"      • Closing other applications")
            print(f"      • Using memory_efficient=True")
            return False
        elif available_gb < 4:
            print(f"   🟡 Moderate memory available")
            return True
        else:
            print(f"   ✅ Sufficient memory available")
            return True

    def analyze_data_quality(self, df):
        """
        Analyze data quality and missing values

        Args:
            df (pd.DataFrame): Input dataset
        """
        print(f"\n🔍 DATA QUALITY ANALYSIS")
        print("=" * 50)

        # Check for required columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = [col for col in self.feature_columns if col not in df.columns]

        print(f"✅ Available features: {available_features}")
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")

        # Check target column
        if self.target_column in df.columns:
            print(f"✅ Target column '{self.target_column}' found")

            # Lithology distribution
            lith_counts = df[self.target_column].value_counts()
            print(f"\n🪨 LITHOLOGY DISTRIBUTION:")
            for lith, count in lith_counts.head(10).items():
                pct = (count / len(df)) * 100
                print(f"   🔸 {lith}: {count:,} samples ({pct:.1f}%)")
        else:
            print(f"❌ Target column '{self.target_column}' not found")
            print(f"Available columns: {list(df.columns)}")

        # Missing value analysis
        print(f"\n📋 MISSING VALUE ANALYSIS:")
        for col in available_features + [self.target_column]:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                status = "✅" if missing_count == 0 else "⚠️" if missing_pct < 10 else "❌"
                print(f"   {status} {col}: {missing_count:,} missing ({missing_pct:.1f}%)")

    def preprocess_data(self, df):
        """
        Enhanced preprocessing pipeline with comprehensive quality tracking

        Args:
            df (pd.DataFrame): Raw dataset

        Returns:
            tuple: (X_processed, y_processed, feature_names)
        """
        print(f"\n🔧 ENHANCED PREPROCESSING WITH QUALITY TRACKING")
        print("=" * 60)

        # Initialize quality tracking system
        self._initialize_quality_tracking(df)

        # Filter for available features
        available_features = [col for col in self.feature_columns if col in df.columns]

        if len(available_features) < 3:
            raise ValueError(f"Insufficient features available. Need at least 3, got {len(available_features)}")

        # Check if target column exists
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        # Remove rows where target is missing
        df_clean = df.dropna(subset=[self.target_column]).copy()
        removed_rows = len(df) - len(df_clean)
        print(f"📊 Removed {removed_rows:,} rows with missing target values")

        # Update quality tracking for removed rows
        if removed_rows > 0:
            removed_indices = df.index.difference(df_clean.index)
            self.quality_annotations = self.quality_annotations.drop(removed_indices)

        # Extract features and target
        X = df_clean[available_features].copy()
        y = df_clean[self.target_column].copy()

        print(f"📈 Dataset shape: {X.shape}")
        print(f"🎯 Target classes: {y.nunique()}")

        # Step 1: Detect and log data quality issues
        null_count = self._detect_and_log_null_values(X, available_features)
        outlier_counts = self._detect_and_log_outliers(X, available_features)

        # Step 2: Test and select best imputation technique
        if null_count > 0:
            best_imputation_technique, best_imputer = self._test_imputation_techniques(X, available_features)
            imputer = best_imputer

            # Apply imputation and log techniques used
            print(f"\n🔧 APPLYING BEST IMPUTATION: {best_imputation_technique.upper()}")
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=available_features,
                index=X.index
            )

            # Log imputation techniques applied
            null_mask = X.isnull().any(axis=1)
            current_techniques = self.quality_annotations.loc[null_mask, 'Techniques_Applied']
            new_techniques = current_techniques.apply(
                lambda x: f"{x}; {best_imputation_technique.capitalize()} Imputation" if x
                else f"{best_imputation_technique.capitalize()} Imputation"
            )
            self.quality_annotations.loc[null_mask, 'Techniques_Applied'] = new_techniques
            self.quality_annotations.loc[null_mask, 'Best_Technique'] = f"Imputation: {best_imputation_technique.capitalize()}"

        else:
            print(f"✅ No missing values detected - skipping imputation")
            X_imputed = X.copy()
            imputer = SimpleImputer(strategy='median')  # Placeholder for consistency

        # Step 3: Handle outliers (log but preserve for geological data)
        outlier_mask = self.quality_annotations['Has_Outliers']
        if outlier_mask.sum() > 0:
            print(f"\n📊 OUTLIER HANDLING: Preserving geological extremes")
            self.quality_annotations.loc[outlier_mask, 'Techniques_Applied'] = \
                self.quality_annotations.loc[outlier_mask, 'Techniques_Applied'].apply(
                    lambda x: f"{x}; Outlier Preserved" if x else "Outlier Preserved"
                )
            self.quality_annotations.loc[outlier_mask, 'Processing_Notes'] = \
                "Outliers preserved for geological validity"

        # Step 4: Feature scaling with technique tracking
        print(f"\n📏 FEATURE SCALING: StandardScaler")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed),
            columns=available_features,
            index=X_imputed.index
        )

        # Log scaling technique for all rows
        current_techniques = self.quality_annotations['Techniques_Applied']
        new_techniques = current_techniques.apply(
            lambda x: f"{x}; StandardScaler" if x else "StandardScaler"
        )
        self.quality_annotations['Techniques_Applied'] = new_techniques

        # Step 5: Normalize target labels (convert mixed types to strings)
        print(f"\n🏷️ LABEL ENCODING WITH MIXED TYPE HANDLING:")
        y_normalized = self._normalize_mixed_data_types(y)

        # Encode the normalized labels
        y_encoded = self.label_encoder.fit_transform(y_normalized)

        print(f"📋 Final label mapping:")
        for i, label in enumerate(self.label_encoder.classes_):
            count = (y_encoded == i).sum()
            print(f"      {label} ({count:,} samples)")

        # Store preprocessing objects
        self.imputer = imputer
        self.scaler = scaler
        self.feature_names = available_features

        # Log preprocessing completion
        self.preprocessing_log.append({
            'timestamp': pd.Timestamp.now(),
            'total_rows': len(X_scaled),
            'features_processed': len(available_features),
            'null_rows_handled': null_count,
            'outlier_rows_detected': outlier_mask.sum(),
            'best_imputation': self.best_techniques.get('imputation', 'None'),
            'scaling_method': 'StandardScaler'
        })

        print(f"\n✅ PREPROCESSING COMPLETED WITH QUALITY TRACKING")
        print(f"📊 Quality annotations ready for {len(self.quality_annotations):,} rows")

        return X_scaled, y_encoded, available_features

    def predict_lithology_names(self, new_data, model_name='best', return_confidence=True):
        """
        Simplified prediction method that guarantees lithology names output

        Args:
            new_data (pd.DataFrame): New well log data
            model_name (str): Model to use ('best', 'random_forest', 'xgboost')
            return_confidence (bool): Whether to return confidence scores

        Returns:
            list or tuple: Lithology names (and confidence scores if requested)
        """
        predictions_dict = self.predict_new_data(new_data, model_name)

        lithology_names = predictions_dict['predictions']

        # Double-check that we have actual names, not numbers
        if len(lithology_names) > 0:
            if isinstance(lithology_names[0], (int, float, np.integer, np.floating)):
                # Force conversion to names
                class_names = list(self.label_encoder.classes_)
                lithology_names = [class_names[int(pred)] if int(pred) < len(class_names)
                                 else f"Unknown_Class_{pred}" for pred in lithology_names]

        if return_confidence:
            confidence_scores = predictions_dict['confidence_scores']
            return lithology_names, confidence_scores
        else:
            return lithology_names

    def get_lithology_mapping(self):
        """
        Get the mapping between numerical labels and lithology names

        Returns:
            dict: Mapping from numbers to lithology names
        """
        if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
            return {i: name for i, name in enumerate(self.label_encoder.classes_)}
        else:
            return {}

    def train_models(self, X, y):
        """
        Train Random Forest and XGBoost models with hyperparameter tuning

        Args:
            X (pd.DataFrame): Preprocessed features
            y (np.array): Encoded target labels
        """
        print(f"\n🤖 MODEL TRAINING")
        print("=" * 50)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📊 Training set: {X_train.shape[0]:,} samples")
        print(f"📊 Test set: {X_test.shape[0]:,} samples")

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        # 1. Random Forest with memory-optimized hyperparameter tuning
        print(f"\n🌲 TRAINING RANDOM FOREST:")

        # Adjust parameters based on dataset size and memory constraints
        dataset_size = len(X_train)

        if dataset_size > 50000 or self.sample_used:
            # Memory-efficient parameters for large datasets
            rf_param_grid = {
                'n_estimators': [50, 100],  # Reduced for memory efficiency
                'max_depth': [10, 15],      # Limited depth to prevent overfitting
                'min_samples_split': [5, 10],  # Higher values for regularization
                'min_samples_leaf': [2, 4]     # Higher values for regularization
            }
            n_jobs = min(4, os.cpu_count() or 1)  # Limit parallel jobs
            cv_folds = 3  # Reduced CV folds for speed
        else:
            # Standard parameters for smaller datasets
            rf_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            n_jobs = -1
            cv_folds = 5

        print(f"   📊 Dataset size: {dataset_size:,} samples")
        print(f"   🧠 Using {cv_folds}-fold CV with {n_jobs} parallel jobs")

        try:
            rf_base = RandomForestClassifier(
                random_state=42,
                n_jobs=1,  # Single job to prevent memory issues
                max_features='sqrt'  # Reduce feature complexity
            )

            rf_grid = GridSearchCV(
                rf_base, rf_param_grid,
                cv=cv_folds,
                scoring='f1_weighted',
                n_jobs=1,  # Single job for GridSearch to prevent memory issues
                verbose=1,
                error_score='raise'  # Raise errors instead of ignoring
            )

            rf_grid.fit(X_train, y_train)
            self.models['random_forest'] = rf_grid.best_estimator_

            print(f"   ✅ Best RF parameters: {rf_grid.best_params_}")
            print(f"   📈 Best CV score: {rf_grid.best_score_:.4f}")

        except Exception as e:
            print(f"   ⚠️  Random Forest training failed: {str(e)}")
            print(f"   🔧 Falling back to simple Random Forest...")

            # Fallback to simple model without hyperparameter tuning
            rf_simple = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
                max_features='sqrt'
            )

            rf_simple.fit(X_train, y_train)
            self.models['random_forest'] = rf_simple
            print(f"   ✅ Simple Random Forest trained successfully")

        # 2. XGBoost with memory-optimized hyperparameter tuning
        print(f"\n🚀 TRAINING XGBOOST:")

        # Adjust XGBoost parameters based on dataset size
        if dataset_size > 50000 or self.sample_used:
            # Memory-efficient parameters for large datasets
            xgb_param_grid = {
                'n_estimators': [50, 100],     # Reduced for memory efficiency
                'max_depth': [3, 6],           # Reasonable depth
                'learning_rate': [0.1, 0.2],   # Standard learning rates
                'subsample': [0.8, 0.9]        # Regularization
            }
        else:
            # Standard parameters for smaller datasets
            xgb_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0]
            }

        try:
            xgb_base = xgb.XGBClassifier(
                random_state=42,
                n_jobs=1,  # Single job to prevent memory issues
                tree_method='hist',  # Memory-efficient tree method
                eval_metric='mlogloss'  # Suppress warnings
            )

            xgb_grid = GridSearchCV(
                xgb_base, xgb_param_grid,
                cv=cv_folds,
                scoring='f1_weighted',
                n_jobs=1,  # Single job for GridSearch
                verbose=1,
                error_score='raise'
            )

            xgb_grid.fit(X_train, y_train)
            self.models['xgboost'] = xgb_grid.best_estimator_

            print(f"   ✅ Best XGB parameters: {xgb_grid.best_params_}")
            print(f"   📈 Best CV score: {xgb_grid.best_score_:.4f}")

        except Exception as e:
            print(f"   ⚠️  XGBoost training failed: {str(e)}")
            print(f"   🔧 Falling back to simple XGBoost...")

            # Fallback to simple model without hyperparameter tuning
            xgb_simple = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                n_jobs=1,
                tree_method='hist',
                eval_metric='mlogloss'
            )

            xgb_simple.fit(X_train, y_train)
            self.models['xgboost'] = xgb_simple
            print(f"   ✅ Simple XGBoost trained successfully")

        print(f"\n✅ Model training completed!")

        # Memory cleanup
        if self.memory_efficient:
            import gc
            gc.collect()
            print(f"🧹 Memory cleanup performed")

    def evaluate_models(self):
        """
        Evaluate trained models and generate comprehensive metrics
        """
        print(f"\n📊 MODEL EVALUATION")
        print("=" * 50)

        self.evaluation_results = {}

        for model_name, model in self.models.items():
            print(f"\n🔍 Evaluating {model_name.upper()}:")

            # Predictions
            y_pred = model.predict(self.X_test)

            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted'
            )

            print(f"   📈 Accuracy: {accuracy:.4f}")
            print(f"   📈 Precision: {precision:.4f}")
            print(f"   📈 Recall: {recall:.4f}")
            print(f"   📈 F1-Score: {f1:.4f}")

            # Store results
            self.evaluation_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'classification_report': classification_report(
                    self.y_test, y_pred,
                    labels=range(len(self.label_encoder.classes_)),
                    target_names=self.label_encoder.classes_,
                    output_dict=True,
                    zero_division=0
                )
            }

            # Detailed classification report
            print(f"\n📋 Classification Report for {model_name.upper()}:")
            try:
                target_names = [str(cls) for cls in self.label_encoder.classes_]
                print(classification_report(
                    self.y_test, y_pred,
                    labels=range(len(self.label_encoder.classes_)),
                    target_names=target_names,
                    zero_division=0
                ))
            except Exception as e:
                print(f"⚠️  Classification report error: {str(e)}")
                print("📊 Basic metrics calculated successfully")

    def create_visualizations(self):
        """
        Create comprehensive visualizations for model interpretation
        """
        print(f"\n🎨 CREATING VISUALIZATIONS")
        print("=" * 50)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Confusion Matrices
        self._plot_confusion_matrices(show_names=True)

        # 2. Feature Importance
        self._plot_feature_importance()

        # 3. Lithology vs Depth plots
        self._plot_lithology_depth()

        # 4. Interactive Plotly visualizations
        self._create_interactive_plots()

        print(f"✅ All visualizations saved to {self.results_dir}/visualizations/")

    def _plot_confusion_matrices(self, show_names=False):
        """Create confusion matrix plots for all models"""
        fig, axes = plt.subplots(1, len(self.models), figsize=(15, 6))
        if len(self.models) == 1:
            axes = [axes]

        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = self.evaluation_results[model_name]['predictions']
            cm = confusion_matrix(self.y_test, y_pred)

            # Use lithology names for labels if requested
            if show_names:
                target_names = [str(cls) for cls in self.label_encoder.classes_]
            else:
                target_names = [str(i) for i in range(len(self.label_encoder.classes_))]

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names,
                ax=axes[idx]
            )

            axes[idx].set_title(f'{model_name.title()} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/confusion_matrices.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        fig, axes = plt.subplots(1, len(self.models), figsize=(15, 6))
        if len(self.models) == 1:
            axes = [axes]

        for idx, (model_name, model) in enumerate(self.models.items()):
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)

                axes[idx].barh(importance_df['feature'], importance_df['importance'])
                axes[idx].set_title(f'{model_name.title()} Feature Importance')
                axes[idx].set_xlabel('Importance')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/feature_importance.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_lithology_depth(self):
        """Create lithology vs depth visualization"""
        if 'DEPTH_MD' not in self.raw_data.columns:
            print("⚠️  DEPTH_MD column not found, skipping depth plots")
            return

        # Get predictions for the best model (highest F1 score)
        best_model_name = max(self.evaluation_results.keys(),
                             key=lambda x: self.evaluation_results[x]['f1_score'])
        best_model = self.models[best_model_name]

        # Create predictions for visualization
        test_indices = self.X_test.index
        test_data = self.raw_data.loc[test_indices].copy()

        if len(test_data) == 0:
            print("⚠️  No test data with depth information available")
            return

        y_pred = self.evaluation_results[best_model_name]['predictions']

        # Decode predictions and actual values
        pred_labels = self.label_encoder.inverse_transform(y_pred)
        actual_labels = self.label_encoder.inverse_transform(self.y_test)

        # Create depth plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

        # Plot 1: Actual vs Predicted
        depths = test_data['DEPTH_MD'].values

        # Create color mapping for lithologies
        unique_lithologies = np.unique(np.concatenate([actual_labels, pred_labels]))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_lithologies)))
        color_map = dict(zip(unique_lithologies, colors))

        # Actual lithology
        for lith in unique_lithologies:
            mask = actual_labels == lith
            if mask.any():
                ax1.scatter(np.ones(mask.sum()) * 1, depths[mask],
                           c=[color_map[lith]], label=lith, alpha=0.7, s=20)

        ax1.set_xlim(0.5, 1.5)
        ax1.set_xlabel('Actual Lithology')
        ax1.set_ylabel('Depth (MD)')
        ax1.set_title('Actual Lithology vs Depth')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.invert_yaxis()

        # Predicted lithology
        for lith in unique_lithologies:
            mask = pred_labels == lith
            if mask.any():
                ax2.scatter(np.ones(mask.sum()) * 1, depths[mask],
                           c=[color_map[lith]], label=lith, alpha=0.7, s=20)

        ax2.set_xlim(0.5, 1.5)
        ax2.set_xlabel('Predicted Lithology')
        ax2.set_ylabel('Depth (MD)')
        ax2.set_title(f'Predicted Lithology vs Depth\n({best_model_name.title()} Model)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/lithology_depth_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_interactive_plots(self):
        """Create interactive Plotly visualizations"""
        # Get the best model
        best_model_name = max(self.evaluation_results.keys(),
                             key=lambda x: self.evaluation_results[x]['f1_score'])

        # 1. Interactive confusion matrix
        y_pred = self.evaluation_results[best_model_name]['predictions']
        cm = confusion_matrix(self.y_test, y_pred)

        # Use lithology names for axes
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=[str(cls) for cls in self.label_encoder.classes_],
            y=[str(cls) for cls in self.label_encoder.classes_],
            color_continuous_scale='Blues',
            title=f'Interactive Confusion Matrix - {best_model_name.title()}'
        )

        fig_cm.update_layout(
            width=800, height=600,
            title_x=0.5,
            font=dict(size=12)
        )

        fig_cm.write_html(f'{self.results_dir}/visualizations/interactive_confusion_matrix.html')

        # 2. Feature importance interactive plot
        if hasattr(self.models[best_model_name], 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.models[best_model_name].feature_importances_
            }).sort_values('Importance', ascending=True)

            fig_imp = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Interactive Feature Importance - {best_model_name.title()}',
                color='Importance',
                color_continuous_scale='Viridis'
            )

            fig_imp.update_layout(
                width=800, height=500,
                title_x=0.5,
                font=dict(size=12)
            )

            fig_imp.write_html(f'{self.results_dir}/visualizations/interactive_feature_importance.html')

        print("📊 Interactive plots saved as HTML files")

    def save_models(self):
        """Save trained models and preprocessing objects"""
        print(f"\n💾 SAVING MODELS AND PREPROCESSING OBJECTS")
        print("=" * 50)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save models
        for model_name, model in self.models.items():
            model_path = f'{self.results_dir}/{model_name}_model_{timestamp}.joblib'
            joblib.dump(model, model_path)
            print(f"✅ Saved {model_name} model: {model_path}")

        # Save preprocessing objects
        preprocessing_objects = {
            'imputer': self.imputer,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }

        preprocessing_path = f'{self.results_dir}/preprocessing_objects_{timestamp}.joblib'
        joblib.dump(preprocessing_objects, preprocessing_path)
        print(f"✅ Saved preprocessing objects: {preprocessing_path}")

        # Save evaluation results
        results_path = f'{self.results_dir}/evaluation_results_{timestamp}.json'

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, results in self.evaluation_results.items():
            json_results[model_name] = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score'])
            }

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"✅ Saved evaluation results: {results_path}")

        return {
            'models': {name: f'{self.results_dir}/{name}_model_{timestamp}.joblib'
                      for name in self.models.keys()},
            'preprocessing': preprocessing_path,
            'results': results_path
        }

    def predict_new_data(self, new_data, model_name='best'):
        """
        Make predictions on new data

        Args:
            new_data (pd.DataFrame): New well log data
            model_name (str): Model to use ('best', 'random_forest', 'xgboost')

        Returns:
            dict: Predictions and probabilities
        """
        if model_name == 'best':
            model_name = max(self.evaluation_results.keys(),
                           key=lambda x: self.evaluation_results[x]['f1_score'])

        model = self.models[model_name]

        # Preprocess new data
        X_new = new_data[self.feature_names].copy()
        X_new_imputed = pd.DataFrame(
            self.imputer.transform(X_new),
            columns=self.feature_names
        )
        X_new_scaled = pd.DataFrame(
            self.scaler.transform(X_new_imputed),
            columns=self.feature_names
        )

        # Make predictions
        predictions = model.predict(X_new_scaled)
        probabilities = model.predict_proba(X_new_scaled)

        # Decode predictions to lithology names - Enhanced with error handling
        try:
            predicted_labels = self.label_encoder.inverse_transform(predictions)
        except Exception as e:
            print(f"⚠️  Warning: Error in label decoding: {e}")
            # Fallback: create mapping manually
            label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
            predicted_labels = [label_mapping.get(pred, f"Unknown_{pred}") for pred in predictions]

        # Ensure we return actual lithology names, not numbers
        if len(predicted_labels) > 0 and isinstance(predicted_labels[0], (int, float, np.integer, np.floating)):
            print("🔧 Converting numerical predictions to lithology names...")
            label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
            predicted_labels = [label_mapping.get(int(pred), f"Unknown_{pred}") for pred in predicted_labels]

        return {
            'predictions': predicted_labels,
            'probabilities': probabilities,
            'class_names': list(self.label_encoder.classes_),
            'model_used': model_name,
            'confidence_scores': np.max(probabilities, axis=1)
        }

    def run_complete_pipeline(self):
        """
        Execute the complete ML pipeline

        Returns:
            dict: Summary of results and saved files
        """
        print("🚀 STARTING COMPLETE LITHOLOGY ML PIPELINE")
        print("=" * 60)

        try:
            # 0. Check memory status
            try:
                memory_ok = self._check_memory_usage()
                if not memory_ok:
                    print("⚠️  Proceeding with caution due to low memory...")
            except ImportError:
                print("💡 Install psutil for memory monitoring: pip install psutil")
            except Exception:
                print("💾 Memory monitoring unavailable, proceeding...")

            # 1. Load and combine data
            df = self.load_and_combine_data()

            # 2. Analyze data quality
            self.analyze_data_quality(df)

            # 3. Preprocess data
            X, y, feature_names = self.preprocess_data(df)

            # 4. Train models
            self.train_models(X, y)

            # 5. Evaluate models
            self.evaluate_models()

            # 6. Create visualizations
            self.create_visualizations()

            # 7. Save models
            saved_files = self.save_models()

            # 8. Create enhanced dataset with quality annotations
            print(f"\n📋 CREATING ENHANCED DATASET WITH QUALITY ANNOTATIONS")

            # Get predictions for the original dataset
            try:
                # Use the best model to predict on the processed data
                best_model_name = max(self.evaluation_results.keys(),
                                     key=lambda x: self.evaluation_results[x]['f1_score'])
                best_model = self.models[best_model_name]

                # Make predictions on the full processed dataset
                predictions = best_model.predict(X)
                prediction_names = self.label_encoder.inverse_transform(predictions)

                # Create enhanced dataset
                enhanced_dataset = self.create_enhanced_dataset_with_annotations(
                    df, predictions=prediction_names
                )

                # Save enhanced dataset
                enhanced_file_path = self.save_enhanced_dataset(enhanced_dataset)
                saved_files['enhanced_dataset'] = enhanced_file_path

            except Exception as e:
                print(f"⚠️ Failed to create enhanced dataset: {str(e)}")
                enhanced_file_path = None

            # 9. Generate summary report
            summary = self._generate_summary_report()

            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📊 Best Model: {summary['best_model']}")
            print(f"📈 Best F1-Score: {summary['best_f1_score']:.4f}")
            print(f"💾 Results saved to: {self.results_dir}")
            if enhanced_file_path:
                print(f"📋 Enhanced dataset: {os.path.basename(enhanced_file_path)}")

            return {
                'summary': summary,
                'saved_files': saved_files,
                'models': self.models,
                'evaluation_results': self.evaluation_results,
                'enhanced_dataset_path': enhanced_file_path
            }

        except Exception as e:
            print(f"❌ Pipeline failed with error: {str(e)}")
            raise

    def _generate_summary_report(self):
        """Generate a summary report of the pipeline results"""
        best_model_name = max(self.evaluation_results.keys(),
                             key=lambda x: self.evaluation_results[x]['f1_score'])
        best_results = self.evaluation_results[best_model_name]

        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.raw_data),
                'features_used': self.feature_names,
                'num_classes': len(self.label_encoder.classes_),
                'class_names': list(self.label_encoder.classes_)
            },
            'best_model': best_model_name,
            'best_f1_score': best_results['f1_score'],
            'all_model_results': {
                name: {
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score']
                }
                for name, results in self.evaluation_results.items()
            }
        }

        # Save summary report
        summary_path = f'{self.results_dir}/pipeline_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


def create_demo_data():
    """
    Create synthetic demo data if no real data is available

    Returns:
        pd.DataFrame: Synthetic well log data with lithology labels
    """
    print("🔧 Creating synthetic demo data...")

    np.random.seed(42)
    n_samples = 5000

    # Define lithology types and their typical log characteristics
    lithologies = {
        'Sandstone': {'GR': (30, 80), 'RHOB': (2.0, 2.4), 'NPHI': (0.05, 0.25),
                     'RDEP': (10, 1000), 'DTC': (80, 120), 'PEF': (1.8, 3.0)},
        'Shale': {'GR': (80, 200), 'RHOB': (2.2, 2.8), 'NPHI': (0.15, 0.45),
                 'RDEP': (1, 20), 'DTC': (100, 200), 'PEF': (2.8, 3.5)},
        'Limestone': {'GR': (10, 60), 'RHOB': (2.4, 2.8), 'NPHI': (0.0, 0.15),
                     'RDEP': (50, 2000), 'DTC': (50, 90), 'PEF': (4.5, 5.5)},
        'Dolomite': {'GR': (10, 50), 'RHOB': (2.6, 2.9), 'NPHI': (0.0, 0.10),
                    'RDEP': (100, 5000), 'DTC': (45, 80), 'PEF': (2.8, 3.2)},
        'Coal': {'GR': (20, 100), 'RHOB': (1.2, 1.8), 'NPHI': (0.25, 0.60),
                'RDEP': (100, 10000), 'DTC': (120, 300), 'PEF': (0.2, 0.8)}
    }

    data = []

    for i in range(n_samples):
        # Random lithology selection
        lith = np.random.choice(list(lithologies.keys()))
        lith_props = lithologies[lith]

        # Generate log values with some noise
        row = {
            'WELL': f'DEMO_WELL_{(i // 1000) + 1}',
            'DEPTH_MD': 1500 + i * 0.5,
            'FORCE_2020_LITHOFACIES_LITHOLOGY': lith
        }

        for log_type, (min_val, max_val) in lith_props.items():
            # Add realistic noise and correlations
            base_value = np.random.uniform(min_val, max_val)
            noise = np.random.normal(0, (max_val - min_val) * 0.05)
            row[log_type] = max(0, base_value + noise)

        data.append(row)

    df = pd.DataFrame(data)

    # Add some missing values to make it realistic
    for col in ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF']:
        missing_mask = np.random.random(len(df)) < 0.02  # 2% missing
        df.loc[missing_mask, col] = np.nan

    return df


def main():
    """
    Main function to run the lithology classification pipeline
    """
    print("🪨 LITHOLOGY CLASSIFICATION ML PIPELINE")
    print("=" * 60)
    print("🎯 Objective: Predict lithology classes from well log data")
    print("🔬 Features: GR, RHOB, NPHI, RDEP, DTC, PEF")
    print("🤖 Models: Random Forest & XGBoost")
    print("=" * 60)

    # Initialize pipeline with memory optimization
    pipeline = LithologyMLPipeline(
        max_samples=100000,  # Limit to 100K samples for memory efficiency
        memory_efficient=True  # Enable memory optimization
    )

    try:
        # Check if real data exists
        if not os.path.exists("litho_data") or not glob.glob("litho_data/*.csv"):
            print("⚠️  No real data found in litho_data directory")
            print("🔧 Creating synthetic demo data...")

            # Create demo data
            demo_df = create_demo_data()
            os.makedirs("litho_data", exist_ok=True)
            demo_df.to_csv("litho_data/demo_synthetic_data.csv", index=False)
            print(f"✅ Created demo data: litho_data/demo_synthetic_data.csv")

        # Run the complete pipeline
        results = pipeline.run_complete_pipeline()

        # Display final summary
        print("\n📋 FINAL SUMMARY")
        print("=" * 40)
        summary = results['summary']
        print(f"🏆 Best Model: {summary['best_model']}")
        print(f"📊 Dataset: {summary['dataset_info']['total_samples']:,} samples")
        print(f"🔢 Features: {len(summary['dataset_info']['features_used'])}")
        print(f"🪨 Lithologies: {summary['dataset_info']['num_classes']}")
        print(f"📈 Best F1-Score: {summary['best_f1_score']:.4f}")

        print(f"\n📁 All results saved to: model_results/")
        print(f"🎨 Visualizations: model_results/visualizations/")

        # Demonstration of inference
        print(f"\n🔮 DEMONSTRATION: Predicting lithology on new data")
        if 'demo_synthetic_data.csv' in glob.glob("litho_data/*.csv"):
            demo_data = pd.read_csv("litho_data/demo_synthetic_data.csv").head(10)
            predictions = pipeline.predict_new_data(demo_data)

            print(f"📊 Lithology Predictions (showing actual rock types):")
            print(f"{'Sample':<8} {'Predicted Lithology':<20} {'Confidence':<12} {'Actual Lithology':<20}")
            print("-" * 65)

            for i, (pred, conf) in enumerate(zip(predictions['predictions'],
                                               predictions['confidence_scores'])):
                actual = demo_data.iloc[i]['FORCE_2020_LITHOFACIES_LITHOLOGY'] if 'FORCE_2020_LITHOFACIES_LITHOLOGY' in demo_data.columns else 'N/A'
                print(f"{i+1:<8} {pred:<20} {conf:.3f}        {actual:<20}")

            print(f"\n✅ All predictions returned as lithology names (not numbers!)")
            print(f"🎯 Available lithology classes: {', '.join(predictions['class_names'])}")
            print(f"🤖 Model used: {predictions['model_used']}")

        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"💡 Ready for presentation to project mentor!")

        return results

    except Exception as e:
        print(f"❌ Error in pipeline execution: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
