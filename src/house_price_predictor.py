"""
Base House Price Predictor with common functionality

Author: Python ML Project
Description: Base class for house price prediction
"""

import pandas as pd
import numpy as np
import joblib          
import json            
from datetime import datetime  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
from clean_data import load_and_clean_data, prepare_features


warnings.filterwarnings('ignore')

class BaseHousePricePredictor:
    """
    Base class for house price prediction with common functionality
    """
    
    def __init__(self, use_advanced_features=True):
        self.use_advanced_features = use_advanced_features
        self.is_trained = False
        
        # Initialize encoders and scalers
        self.location_encoder = LabelEncoder()
        self.area_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Store feature columns for prediction
        self.feature_columns = None
        self.model = None  # To be set by child classes

    def save_model(self, filepath, include_metadata=True):
        """Save model with metadata"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save the model
        joblib.dump(self.model, f"{filepath}_model.pkl")
        # Save encoders and scalers
        joblib.dump(self.location_encoder, f"{filepath}_location_encoder.pkl")
        joblib.dump(self.area_encoder, f"{filepath}_area_encoder.pkl")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        if include_metadata:
            metadata = {
                'model_type': getattr(self, 'model_type', 'unknown'),
                'use_advanced_features': self.use_advanced_features,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'training_date': datetime.now().isoformat(),
                'model_params': self.model.get_params() if self.model else {}
            }
            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved successfully to {filepath}")

    def load_model(self, filepath):
        """Load model with metadata"""
        try:
            self.model = joblib.load(f"{filepath}_model.pkl")
            self.location_encoder = joblib.load(f"{filepath}_location_encoder.pkl")
            self.area_encoder = joblib.load(f"{filepath}_area_encoder.pkl")
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.is_trained = metadata['is_trained']
            self.use_advanced_features = metadata['use_advanced_features']
            
            print(f"✅ Model loaded successfully from {filepath}")
            print(f"   Training date: {metadata.get('training_date', 'Unknown')}")
            
        except FileNotFoundError as e:
            print(f"❌ Model files not found at {filepath}. Error: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_and_clean_data(self, csv_file_path):
        """Load and clean the dataset with advanced processing"""
        return load_and_clean_data(
            csv_file_path, 
            self.use_advanced_features, 
            self.location_encoder, 
            self.area_encoder
        )
    
    def prepare_features(self, df):
        """Prepare features for training (basic or advanced)"""
        if self.use_advanced_features:
            X, y, feature_columns = prepare_features(df, True, self.scaler)
        else:
            X, y, feature_columns = prepare_features(df, False)
        
        self.feature_columns = feature_columns
        return X, y
    
    def predict_price(self, bhk, sqft, bathrooms, location='Unknown', area_type='Unknown'):
        """Predict house price for given features"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please call train_model() first.")
        
        # Validate inputs
        if bhk <= 0 or sqft <= 0 or bathrooms <= 0:
            raise ValueError("All features must be positive values")
        
        if self.use_advanced_features:
            # Create advanced features for prediction
            # Calculate derived features
            bath_per_bhk = bathrooms / bhk
            sqft_per_bhk = sqft / bhk
            
            # Encode location (handle unknown locations)
            try:
                location_encoded = self.location_encoder.transform([location])[0]
            except ValueError:
                # If location not seen during training, use most common location
                location_encoded = 0
            
            # Encode area type
            try:
                area_type_encoded = self.area_encoder.transform([area_type])[0]
            except ValueError:
                area_type_encoded = 0
            
            # Use average location stats for unknown locations
            location_price_mean = 75  # Average price
            location_price_median = 65  # Median price
            
            # Create feature vector
            features = [
                bhk, sqft, bathrooms, 0,  # Balcony default to 0
                bath_per_bhk, sqft_per_bhk,
                location_encoded, area_type_encoded,
                location_price_mean, location_price_median
            ]
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)
        else:
            # Basic prediction with just 3 features
            prediction = self.model.predict([[bhk, sqft, bathrooms]])
        
        return prediction[0]
    
    def predict_multiple(self, feature_list):
        """Predict prices for multiple houses"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please call train_model() first.")
        
        predictions = []
        for features in feature_list:
            if len(features) == 3:
                bhk, sqft, bathrooms = features
                price = self.predict_price(bhk, sqft, bathrooms)
            elif len(features) == 5:
                bhk, sqft, bathrooms, location, area_type = features
                price = self.predict_price(bhk, sqft, bathrooms, location, area_type)
            else:
                raise ValueError("Each feature tuple must have 3 or 5 elements")
            predictions.append(price)
        
        return predictions
    
    # Abstract method to be implemented by child classes
    def train_model(self, X, y):
        raise NotImplementedError("Child classes must implement train_model method")
