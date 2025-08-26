from random_forest import RandomForestPredictor
import os

def test_model_persistence(save_directory="../notebooks"):
    """Test model persistence with configurable save directory"""
    
    # Test the persistence functionality
    predictor = RandomForestPredictor(use_advanced_features=True)
    
    try:
        print(f"🧪 Testing model persistence...")
        print(f"📁 Save directory: {os.path.abspath(save_directory)}")
        
        # Load and train
        print("📊 Loading and training model...")
        df_clean = predictor.load_and_clean_data("../data/Bengaluru_House_Data_ML_Project.csv")
        X, y = predictor.prepare_features(df_clean)
        predictor.train_model(X, y)
        
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Test saving with full path
        save_path = os.path.join(save_directory, "my_test_model")
        print(f"💾 Saving model to: {save_path}")
        predictor.save_model(save_path)
        print("✅ SUCCESS: Model saved!")
        
        # List and verify files created
        files = [f for f in os.listdir(save_directory) if f.startswith('my_test_model')]
        print(f"\n📋 Files created ({len(files)} total):")
        for file in sorted(files):
            file_path = os.path.join(save_directory, file)
            file_size = os.path.getsize(file_path)
            print(f"  ✓ {file} ({file_size:,} bytes)")
        
        # Test loading to verify persistence works
        print(f"\n🔄 Testing model loading...")
        new_predictor = RandomForestPredictor(use_advanced_features=True)
        new_predictor.load_model(save_path)
        print("✅ SUCCESS: Model loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

# Run the test
if __name__ == "__main__":
    success = test_model_persistence("../notebooks")
    if success:
        print("\n🎉 All persistence tests passed!")
    else:
        print("\n💥 Persistence tests failed!")
