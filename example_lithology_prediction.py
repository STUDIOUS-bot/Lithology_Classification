"""
🪨 EXAMPLE: How to Use Lithology Classification Pipeline
=====================================================

This script demonstrates how to use the enhanced lithology classification pipeline
to get actual lithology names (Sandstone, Shale, etc.) instead of numerical codes.
"""

import pandas as pd
import numpy as np
from lithology_ml_pipeline import LithologyMLPipeline

def create_sample_well_data():
    """Create sample well log data for demonstration"""
    
    # Sample well log data with realistic values
    well_data = pd.DataFrame({
        'WELL': ['DEMO_WELL_1'] * 10,
        'DEPTH_MD': np.linspace(2000, 2050, 10),
        
        # Well log parameters with realistic ranges
        'GR': [45, 120, 35, 80, 150, 25, 90, 60, 110, 40],      # Gamma Ray (API)
        'RHOB': [2.3, 2.6, 2.2, 2.5, 2.7, 2.1, 2.4, 2.3, 2.6, 2.2],  # Bulk Density (g/cc)
        'NPHI': [0.15, 0.35, 0.10, 0.25, 0.40, 0.08, 0.20, 0.18, 0.30, 0.12],  # Neutron Porosity
        'RDEP': [50, 5, 80, 15, 3, 100, 20, 40, 8, 60],         # Deep Resistivity (Ohm·m)
        'DTC': [70, 95, 65, 85, 100, 60, 80, 75, 90, 68],       # Sonic Transit Time (μs/ft)
        'PEF': [2.8, 3.2, 2.5, 3.0, 3.5, 2.2, 2.9, 2.7, 3.1, 2.6]  # Photoelectric Factor
    })
    
    return well_data

def demonstrate_lithology_prediction():
    """Demonstrate lithology prediction with actual names"""
    
    print("🪨 LITHOLOGY PREDICTION DEMONSTRATION")
    print("=" * 50)
    
    # Step 1: Create sample data
    print("📊 Step 1: Creating sample well log data...")
    sample_data = create_sample_well_data()
    print(f"   ✅ Created {len(sample_data)} data points")
    print(f"   📋 Features: {', '.join(['GR', 'RHOB', 'NPHI', 'RDEP', 'DTC', 'PEF'])}")
    
    # Display sample data
    print(f"\n📋 Sample Well Log Data:")
    print(sample_data[['DEPTH_MD', 'GR', 'RHOB', 'NPHI', 'RDEP']].head())
    
    # Step 2: Initialize pipeline
    print(f"\n🤖 Step 2: Initializing ML Pipeline...")
    pipeline = LithologyMLPipeline()
    
    # Note: In real usage, you would train the models first or load pre-trained models
    print("💡 Note: This assumes you have already trained models using:")
    print("   python lithology_ml_pipeline.py")
    
    try:
        # Step 3: Load trained models (if available)
        print(f"\n📂 Step 3: Loading trained models...")
        
        # This would load your pre-trained models
        # For demonstration, we'll show the expected workflow
        
        # Step 4: Make predictions
        print(f"\n🔮 Step 4: Making lithology predictions...")
        
        # Method 1: Full prediction with probabilities
        print(f"\n📊 Method 1: Full prediction results")
        print("predictions = pipeline.predict_new_data(sample_data)")
        print("Expected output format:")
        print({
            'predictions': ['Sandstone', 'Shale', 'Sandstone', 'Shale', 'Limestone'],
            'probabilities': 'Array of probability scores for each class',
            'class_names': ['Sandstone', 'Shale', 'Limestone', 'Dolomite', 'Anhydrite'],
            'model_used': 'xgboost',
            'confidence_scores': [0.85, 0.92, 0.78, 0.88, 0.91]
        })
        
        # Method 2: Simplified prediction (names only)
        print(f"\n📝 Method 2: Simplified prediction (names only)")
        print("lithology_names = pipeline.predict_lithology_names(sample_data, return_confidence=False)")
        print("Expected output: ['Sandstone', 'Shale', 'Sandstone', 'Shale', 'Limestone']")
        
        # Method 3: Names with confidence scores
        print(f"\n🎯 Method 3: Names with confidence scores")
        print("names, confidence = pipeline.predict_lithology_names(sample_data, return_confidence=True)")
        print("Expected output:")
        print("  names: ['Sandstone', 'Shale', 'Sandstone', 'Shale', 'Limestone']")
        print("  confidence: [0.85, 0.92, 0.78, 0.88, 0.91]")
        
        # Step 5: Understanding the results
        print(f"\n📋 Step 5: Understanding the Results")
        print("✅ Predictions are returned as actual lithology names:")
        print("   • 'Sandstone' - Clean reservoir rock")
        print("   • 'Shale' - Fine-grained seal rock") 
        print("   • 'Limestone' - Carbonate reservoir rock")
        print("   • 'Dolomite' - Carbonate with enhanced porosity")
        print("   • 'Anhydrite' - Evaporite tight formation")
        print("   • And more...")
        
        print(f"\n❌ NO MORE numerical codes like:")
        print("   • 65000, 70000, 30000 (confusing numbers)")
        print("   • Users get clear, interpretable results!")
        
        # Step 6: Get lithology mapping
        print(f"\n🗺️  Step 6: Get lithology class mapping")
        print("mapping = pipeline.get_lithology_mapping()")
        print("Expected output: {0: 'Sandstone', 1: 'Shale', 2: 'Limestone', ...}")
        
    except Exception as e:
        print(f"⚠️  Note: {str(e)}")
        print("💡 To run this example with real predictions:")
        print("   1. First train the models: python lithology_ml_pipeline.py")
        print("   2. Then run this example script")

def show_usage_patterns():
    """Show common usage patterns"""
    
    print(f"\n💡 COMMON USAGE PATTERNS")
    print("=" * 30)
    
    print(f"\n1️⃣  Basic Prediction:")
    print("""
# Load your well log data
well_data = pd.read_csv('your_well_data.csv')

# Initialize and use pipeline
pipeline = LithologyMLPipeline()
results = pipeline.predict_new_data(well_data)

# Get lithology names
lithology_names = results['predictions']
print(lithology_names)  # ['Sandstone', 'Shale', 'Limestone', ...]
""")
    
    print(f"\n2️⃣  Batch Processing:")
    print("""
# Process multiple wells
well_files = ['well1.csv', 'well2.csv', 'well3.csv']

for well_file in well_files:
    data = pd.read_csv(well_file)
    predictions = pipeline.predict_lithology_names(data)
    
    # Save results
    results_df = data.copy()
    results_df['Predicted_Lithology'] = predictions
    results_df.to_csv(f'results_{well_file}', index=False)
""")
    
    print(f"\n3️⃣  Real-time Prediction:")
    print("""
# For real-time drilling applications
def predict_current_lithology(gr, rhob, nphi, rdep, dtc, pef):
    current_data = pd.DataFrame({
        'GR': [gr], 'RHOB': [rhob], 'NPHI': [nphi],
        'RDEP': [rdep], 'DTC': [dtc], 'PEF': [pef]
    })
    
    lithology_name = pipeline.predict_lithology_names(current_data)[0]
    return lithology_name

# Usage
current_lithology = predict_current_lithology(75, 2.4, 0.2, 25, 80, 2.8)
print(f"Current formation: {current_lithology}")  # "Sandstone"
""")

def main():
    """Main demonstration function"""
    
    print("🎯 ENHANCED LITHOLOGY CLASSIFICATION PIPELINE")
    print("=" * 60)
    print("🔥 NEW FEATURE: Returns actual lithology names instead of numbers!")
    print("=" * 60)
    
    # Run demonstration
    demonstrate_lithology_prediction()
    
    # Show usage patterns
    show_usage_patterns()
    
    print(f"\n🎉 SUMMARY")
    print("=" * 20)
    print("✅ Pipeline enhanced to return lithology NAMES")
    print("✅ Multiple prediction methods available")
    print("✅ Easy integration with existing workflows")
    print("✅ Clear, interpretable results for geologists")
    
    print(f"\n🚀 Ready to use! Train your models and start predicting!")
    print("=" * 60)

if __name__ == "__main__":
    main()
