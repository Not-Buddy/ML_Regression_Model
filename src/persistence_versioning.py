import joblib
import json
from datetime import datetime

def save_model(self, filepath, include_metadata=True):
    """Save model with metadata"""
    # Save the model
    joblib.dump(self.model, f"{filepath}_model.pkl")
    
    # Save encoders and scalers
    joblib.dump(self.location_encoder, f"{filepath}_location_encoder.pkl")
    joblib.dump(self.area_encoder, f"{filepath}_area_encoder.pkl")
    joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
    
    if include_metadata:
        metadata = {
            'model_type': self.model_type,
            'use_advanced_features': self.use_advanced_features,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'training_date': datetime.now().isoformat(),
            'model_params': self.model.get_params()
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

def load_model(self, filepath):
    """Load model with metadata"""
    # Load the model
    self.model = joblib.load(f"{filepath}_model.pkl")
    
    # Load encoders and scalers
    self.location_encoder = joblib.load(f"{filepath}_location_encoder.pkl")
    self.area_encoder = joblib.load(f"{filepath}_area_encoder.pkl")
    self.scaler = joblib.load(f"{filepath}_scaler.pkl")
    
    # Load metadata
    with open(f"{filepath}_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    self.feature_columns = metadata['feature_columns']
    self.is_trained = metadata['is_trained']
    self.use_advanced_features = metadata['use_advanced_features']
