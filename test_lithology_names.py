"""
TEST SCRIPT: Verify Lithology Names Output
===========================================

This script tests that the lithology ML pipeline returns actual lithology names
(like "Sandstone", "Shale") instead of numerical codes (like 65000, 70000).
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

def create_test_data():
    """Create sample well log data for testing"""
    np.random.seed(42)
    
    # Create realistic well log data
    n_samples = 20
    data = {
        'WELL': [f'TEST_WELL_{i//10 + 1}' for i in range(n_samples)],
        'DEPTH_MD': np.linspace(2000, 2100, n_samples),
        'GR': np.random.uniform(20, 150, n_samples),      # Gamma Ray
        'RHOB': np.random.uniform(2.0, 2.8, n_samples),   # Bulk Density
        'NPHI': np.random.uniform(0.05, 0.4, n_samples),  # Neutron Porosity
        'RDEP': np.random.uniform(1, 100, n_samples),     # Deep Resistivity
        'DTC': np.random.uniform(60, 120, n_samples),     # Sonic Transit Time
        'PEF': np.random.uniform(1.5, 5.0, n_samples)     # Photoelectric Factor
    }
    
    return pd.DataFrame(data)

def test_lithology_prediction():
    """Test that predictions return lithology names"""
    print("🧪 TESTING LITHOLOGY NAME PREDICTION")
    print("=" * 50)
    
    try:
        # Import the pipeline
        from lithology_ml_pipeline import LithologyMLPipeline
        
        # Check if trained models exist
        if not os.path.exists("model_results"):
            print("❌ No trained models found!")
            print("💡 Please run the training pipeline first:")
            print("   python lithology_ml_pipeline.py")
            return False
        
        # Initialize pipeline
        pipeline = LithologyMLPipeline()
        
        # Load trained models
        print("📂 Loading trained models...")
        model_files = [f for f in os.listdir("model_results") if f.endswith('.joblib') and 'model' in f]
        
        if not model_files:
            print("❌ No model files found in model_results/")
            print("💡 Please run the training pipeline first:")
            print("   python lithology_ml_pipeline.py")
            return False
        
        # Load models and preprocessing objects
        import joblib
        
        # Load preprocessing objects
        preprocessing_files = [f for f in os.listdir("model_results") if 'preprocessing' in f]
        if preprocessing_files:
            preprocessing_path = os.path.join("model_results", preprocessing_files[0])
            preprocessing_objects = joblib.load(preprocessing_path)
            
            pipeline.label_encoder = preprocessing_objects['label_encoder']
            pipeline.scaler = preprocessing_objects['scaler']
            pipeline.imputer = preprocessing_objects['imputer']
            pipeline.feature_names = preprocessing_objects['feature_names']
        
        # Load models
        for model_file in model_files:
            model_path = os.path.join("model_results", model_file)
            model = joblib.load(model_path)
            
            if 'random_forest' in model_file:
                pipeline.models['random_forest'] = model
            elif 'xgboost' in model_file:
                pipeline.models['xgboost'] = model
        
        # Create dummy evaluation results for best model selection
        pipeline.evaluation_results = {
            'random_forest': {'f1_score': 0.85},
            'xgboost': {'f1_score': 0.90}
        }
        
        # Create test data
        print("📊 Creating test well log data...")
        test_data = create_test_data()
        print(f"   ✅ Created {len(test_data)} test samples")
        
        # Make predictions
        print("\n🔮 Making lithology predictions...")
        predictions = pipeline.predict_new_data(test_data)
        
        # Display results
        print(f"\n📋 PREDICTION RESULTS:")
        print(f"{'Sample':<8} {'Predicted Lithology':<20} {'Confidence':<12} {'Type Check':<15}")
        print("-" * 60)
        
        all_names = True
        for i, (pred, conf) in enumerate(zip(predictions['predictions'], 
                                           predictions['confidence_scores'])):
            # Check if prediction is a name or number
            is_name = isinstance(pred, str) and not pred.isdigit()
            type_check = "✅ NAME" if is_name else "❌ NUMBER"
            
            if not is_name:
                all_names = False
            
            print(f"{i+1:<8} {pred:<20} {conf:.3f}        {type_check:<15}")
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   🎯 Total predictions: {len(predictions['predictions'])}")
        print(f"   📝 Available classes: {len(predictions['class_names'])}")
        print(f"   🏷️  Class names: {', '.join(predictions['class_names'])}")
        print(f"   🤖 Model used: {predictions['model_used']}")
        
        if all_names:
            print(f"\n✅ SUCCESS: All predictions returned as lithology NAMES!")
            print(f"🎉 The pipeline correctly converts numerical predictions to rock type names.")
        else:
            print(f"\n❌ ISSUE: Some predictions returned as numbers instead of names.")
            print(f"🔧 The pipeline needs debugging for proper name conversion.")
        
        # Test the simplified prediction method
        print(f"\n🧪 Testing simplified prediction method...")
        simple_predictions = pipeline.predict_lithology_names(test_data.head(5))
        lithology_names, confidence_scores = simple_predictions
        
        print(f"📋 Simplified method results:")
        for i, (name, conf) in enumerate(zip(lithology_names, confidence_scores)):
            print(f"   Sample {i+1}: {name} (confidence: {conf:.3f})")
        
        # Test lithology mapping
        print(f"\n🗺️  Lithology mapping:")
        mapping = pipeline.get_lithology_mapping()
        for num, name in mapping.items():
            print(f"   {num}: {name}")
        
        return all_names
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the lithology name test"""
    print("🪨 LITHOLOGY CLASSIFICATION - NAME VERIFICATION TEST")
    print("=" * 60)
    
    success = test_lithology_prediction()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST PASSED: Pipeline returns lithology names correctly!")
        print("✅ Users will see rock types like 'Sandstone', 'Shale', etc.")
        print("✅ No more numerical codes like 65000, 70000")
    else:
        print("⚠️  TEST ISSUES: Pipeline may need debugging")
        print("💡 Check the error messages above for details")
    
    print("\n💡 USAGE EXAMPLES:")
    print("# Basic prediction")
    print("predictions = pipeline.predict_new_data(your_data)")
    print("lithology_names = predictions['predictions']")
    print()
    print("# Simplified method")
    print("names, confidence = pipeline.predict_lithology_names(your_data)")
    print()
    print("# Get lithology mapping")
    print("mapping = pipeline.get_lithology_mapping()")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
