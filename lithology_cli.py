"""
🪨 LITHOLOGY CLASSIFICATION CLI
==============================
Command-line interface for lithology classification using trained ML models.

Usage:
    python lithology_cli.py --input data.csv --model random_forest --output predictions.csv

Features:
- Batch prediction on CSV files
- Model selection
- Confidence thresholding
- Summary statistics

Author: ONGC Petrophysical Analysis Team
Date: 2025-01-19
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import os
import glob
from datetime import datetime
import json

class LithologyCLI:
    def __init__(self):
        self.models = {}
        self.preprocessing_objects = None
        self.feature_columns = ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF']

    def load_models(self):
        """Load trained models and preprocessing objects"""
        print("🔄 Loading trained models...")

        model_files = glob.glob("model_results/*_model_*.joblib")
        preprocessing_files = glob.glob("model_results/preprocessing_objects_*.joblib")

        if not model_files or not preprocessing_files:
            raise FileNotFoundError("❌ No trained models found! Run the training pipeline first.")

        # Load latest models
        latest_timestamp = max([f.split('_')[-1].replace('.joblib', '') for f in model_files])

        for model_file in model_files:
            if latest_timestamp in model_file:
                model_name = model_file.split('/')[-1].split('_model_')[0]
                if os.name == 'nt':  # Windows
                    model_name = model_file.split('\\')[-1].split('_model_')[0]
                self.models[model_name] = joblib.load(model_file)
                print(f"   ✅ Loaded {model_name} model")

        # Load preprocessing objects
        latest_preprocessing = max(preprocessing_files, key=os.path.getctime)
        self.preprocessing_objects = joblib.load(latest_preprocessing)
        print(f"   ✅ Loaded preprocessing objects")

        return True

    def preprocess_data(self, df):
        """Preprocess input data"""
        available_features = [col for col in self.feature_columns if col in df.columns]

        if len(available_features) < 3:
            raise ValueError(f"Insufficient features. Need at least 3, got {len(available_features)}")

        print(f"📊 Using features: {available_features}")

        X = df[available_features].copy()

        # Apply preprocessing
        X_imputed = pd.DataFrame(
            self.preprocessing_objects['imputer'].transform(X),
            columns=available_features,
            index=X.index
        )

        X_scaled = pd.DataFrame(
            self.preprocessing_objects['scaler'].transform(X_imputed),
            columns=available_features,
            index=X_imputed.index
        )

        return X_scaled, available_features

    def predict(self, df, model_name='random_forest', confidence_threshold=0.0):
        """Make predictions on input data"""
        print(f"🔮 Making predictions with {model_name} model...")

        X_processed, features = self.preprocess_data(df)

        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available_models}")

        model = self.models[model_name]
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)

        # Decode predictions
        label_encoder = self.preprocessing_objects['label_encoder']
        predicted_labels = label_encoder.inverse_transform(predictions)

        # Apply confidence threshold
        max_probs = np.max(probabilities, axis=1)
        low_confidence_mask = max_probs < confidence_threshold

        if confidence_threshold > 0 and low_confidence_mask.any():
            predicted_labels[low_confidence_mask] = 'LOW_CONFIDENCE'
            print(f"⚠️  {low_confidence_mask.sum()} predictions below confidence threshold ({confidence_threshold})")

        return {
            'predictions': predicted_labels,
            'probabilities': probabilities,
            'max_probabilities': max_probs,
            'class_names': label_encoder.classes_,
            'features_used': features
        }

    def save_predictions(self, df, results, output_path):
        """Save predictions to CSV file"""
        print(f"💾 Saving predictions to {output_path}...")

        # Create output dataframe
        df_output = df.copy()
        df_output['Predicted_Lithology'] = results['predictions']
        df_output['Prediction_Confidence'] = results['max_probabilities']

        # Add probability columns for each class
        for i, class_name in enumerate(results['class_names']):
            df_output[f'Prob_{class_name}'] = results['probabilities'][:, i]

        # Save to CSV
        df_output.to_csv(output_path, index=False)
        print(f"✅ Predictions saved successfully!")

        return df_output

    def print_summary(self, results, df_output):
        """Print prediction summary statistics"""
        print(f"\n📊 PREDICTION SUMMARY")
        print("=" * 40)

        predictions = results['predictions']
        confidences = results['max_probabilities']

        print(f"📈 Total samples: {len(predictions):,}")
        print(f"📈 Average confidence: {np.mean(confidences):.3f}")
        print(f"📈 Min confidence: {np.min(confidences):.3f}")
        print(f"📈 Max confidence: {np.max(confidences):.3f}")

        print(f"\n🪨 LITHOLOGY DISTRIBUTION:")
        lith_counts = pd.Series(predictions).value_counts()
        for lith, count in lith_counts.items():
            pct = (count / len(predictions)) * 100
            print(f"   🔸 {lith}: {count:,} samples ({pct:.1f}%)")

        print(f"\n📊 CONFIDENCE DISTRIBUTION:")
        confidence_bins = pd.cut(confidences, bins=[0, 0.5, 0.7, 0.9, 1.0],
                               labels=['Low (0-0.5)', 'Medium (0.5-0.7)',
                                      'High (0.7-0.9)', 'Very High (0.9-1.0)'])
        conf_counts = confidence_bins.value_counts()
        for conf_level, count in conf_counts.items():
            pct = (count / len(predictions)) * 100
            print(f"   📊 {conf_level}: {count:,} samples ({pct:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description="🪨 Lithology Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lithology_cli.py --input data.csv --output predictions.csv
  python lithology_cli.py --input data.csv --model xgboost --confidence 0.7
  python lithology_cli.py --input data.csv --list-models
        """
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input CSV file with well log data')
    parser.add_argument('--output', '-o', type=str,
                       help='Output CSV file for predictions (default: auto-generated)')
    parser.add_argument('--model', '-m', type=str, default='random_forest',
                       help='Model to use for predictions (default: random_forest)')
    parser.add_argument('--confidence', '-c', type=float, default=0.0,
                       help='Minimum confidence threshold (default: 0.0)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print summary, do not save predictions')
    parser.add_argument('--column-mapping', type=str,
                       help='Manual column mapping in format "old1:new1,old2:new2"')
    parser.add_argument('--show-columns', action='store_true',
                       help='Show available columns and exit')

    args = parser.parse_args()

    print("🪨 LITHOLOGY CLASSIFICATION CLI")
    print("=" * 40)

    try:
        # Initialize CLI
        cli = LithologyCLI()
        cli.load_models()

        # List models if requested
        if args.list_models:
            print(f"\n🤖 AVAILABLE MODELS:")
            for model_name in cli.models.keys():
                print(f"   • {model_name}")
            return

        # Load input data with enhanced diagnostics
        print(f"\n📁 Loading input data: {args.input}")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")

        # Try different delimiters and encodings
        df = None
        delimiters = [',', ';', '\t', '|']
        encodings = ['utf-8', 'latin-1', 'cp1252']

        for delimiter in delimiters:
            for encoding in encodings:
                try:
                    df = pd.read_csv(args.input, delimiter=delimiter, encoding=encoding)
                    if len(df.columns) > 1:  # Found valid format
                        print(f"✅ Loaded {len(df):,} samples with {len(df.columns)} columns")
                        print(f"   📋 Using delimiter: '{delimiter}', encoding: {encoding}")
                        break
                except:
                    continue
            if df is not None and len(df.columns) > 1:
                break

        if df is None or len(df.columns) <= 1:
            # Fallback to default CSV reading
            df = pd.read_csv(args.input)
            print(f"⚠️  Loaded {len(df):,} samples with {len(df.columns)} columns")

        # Display column information
        print(f"\n📋 COLUMN ANALYSIS:")
        print(f"   Available columns: {list(df.columns)}")

        # Check for required features
        required_features = ['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF']
        available_features = [col for col in required_features if col in df.columns]
        missing_features = [col for col in required_features if col not in df.columns]

        print(f"   ✅ Found features: {available_features}")
        if missing_features:
            print(f"   ❌ Missing features: {missing_features}")

        # Suggest column mapping if needed
        if len(available_features) == 0:
            print(f"\n💡 COLUMN MAPPING SUGGESTIONS:")
            column_mapping = {
                'gamma': 'GR', 'gamma_ray': 'GR', 'gr': 'GR',
                'density': 'RHOB', 'bulk_density': 'RHOB', 'rhob': 'RHOB', 'den': 'RHOB',
                'neutron': 'NPHI', 'neutron_porosity': 'NPHI', 'nphi': 'NPHI', 'neu': 'NPHI',
                'resistivity': 'RDEP', 'deep_resistivity': 'RDEP', 'rdep': 'RDEP', 'res': 'RDEP',
                'dt': 'DTC', 'delta_time': 'DTC', 'dtc': 'DTC', 'sonic': 'DTC',
                'pe': 'PEF', 'photoelectric': 'PEF', 'pef': 'PEF'
            }

            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in column_mapping:
                    print(f"   🔄 '{col}' → '{column_mapping[col_lower]}'")

        # Show first few rows for inspection
        print(f"\n📊 FIRST 5 ROWS:")
        print(df.head().to_string())

        # Handle show-columns argument
        if args.show_columns:
            print(f"\n📋 AVAILABLE COLUMNS IN YOUR FILE:")
            for i, col in enumerate(df.columns, 1):
                print(f"   {i}. {col}")
            print(f"\n💡 Use --column-mapping to map columns, e.g.:")
            print(f"   --column-mapping \"your_column1:GR,your_column2:RHOB\"")
            return

        # Handle manual column mapping
        if args.column_mapping:
            print(f"\n🔄 APPLYING MANUAL COLUMN MAPPING:")
            mapping_pairs = args.column_mapping.split(',')
            rename_dict = {}
            for pair in mapping_pairs:
                if ':' in pair:
                    old_name, new_name = pair.split(':', 1)
                    old_name, new_name = old_name.strip(), new_name.strip()
                    if old_name in df.columns:
                        rename_dict[old_name] = new_name
                        print(f"   🔄 Mapping '{old_name}' → '{new_name}'")
                    else:
                        print(f"   ❌ Column '{old_name}' not found in file")

            if rename_dict:
                df = df.rename(columns=rename_dict)
                available_features = [col for col in required_features if col in df.columns]
                print(f"   ✅ After manual mapping, found features: {available_features}")

        # Auto-map columns if possible and no manual mapping provided
        elif len(available_features) == 0:
            print(f"\n🔄 ATTEMPTING AUTOMATIC COLUMN MAPPING:")
            column_mapping = {
                'gamma': 'GR', 'gamma_ray': 'GR', 'gr': 'GR',
                'density': 'RHOB', 'bulk_density': 'RHOB', 'rhob': 'RHOB', 'den': 'RHOB',
                'neutron': 'NPHI', 'neutron_porosity': 'NPHI', 'nphi': 'NPHI', 'neu': 'NPHI',
                'resistivity': 'RDEP', 'deep_resistivity': 'RDEP', 'rdep': 'RDEP', 'res': 'RDEP',
                'dt': 'DTC', 'delta_time': 'DTC', 'dtc': 'DTC', 'sonic': 'DTC',
                'pe': 'PEF', 'photoelectric': 'PEF', 'pef': 'PEF'
            }

            rename_dict = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in column_mapping:
                    rename_dict[col] = column_mapping[col_lower]
                    print(f"   🔄 Mapping '{col}' → '{column_mapping[col_lower]}'")

            if rename_dict:
                df = df.rename(columns=rename_dict)
                available_features = [col for col in required_features if col in df.columns]
                print(f"   ✅ After mapping, found features: {available_features}")
            else:
                print(f"   ❌ No automatic mapping possible")

        # Make predictions
        results = cli.predict(df, args.model, args.confidence)

        # Generate output filename if not provided
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = f"{base_name}_predictions_{timestamp}.csv"

        # Save predictions (unless summary-only)
        if not args.summary_only:
            df_output = cli.save_predictions(df, results, args.output)
        else:
            df_output = df.copy()
            df_output['Predicted_Lithology'] = results['predictions']
            df_output['Prediction_Confidence'] = results['max_probabilities']

        # Print summary
        cli.print_summary(results, df_output)

        if not args.summary_only:
            print(f"\n✅ Process completed successfully!")
            print(f"📁 Output saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
