"""
Random Forest implementation for House Price Prediction

Author: Python ML Project
Description: Random Forest specific implementation with advanced features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from house_price_predictor import BaseHousePricePredictor

class RandomForestPredictor(BaseHousePricePredictor):
    """
    Random Forest implementation for house price prediction
    """
    
    def __init__(self, use_advanced_features=True, n_estimators=100, max_depth=10):
        super().__init__(use_advanced_features)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            random_state=42
        )
        self.model_type = 'random_forest'
        self.feature_importance_df = None
    
    def tune_hyperparameters(self, X, y):
        """Automatically find best hyperparameters"""
        print("Tuning hyperparameters...")
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [8, 10, 12],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_model(self, X, y, tune_hyperparameters=False):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Optionally tune hyperparameters
        if tune_hyperparameters and len(X) > 500:  # Only for larger datasets
            self.model = self.tune_hyperparameters(X_train, y_train)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Cross-validation for better evaluation
        if len(X) > 100:  # Only if we have enough data
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
            print(f"Cross-Validation R² Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # Test set evaluation
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nRandom Forest Performance:")
        print(f"Test Set R² Score: {r2:.4f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")
        
        # Feature importance
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Most Important Features:")
        print(self.feature_importance_df.head())
        
        return {
            'r2_score': r2,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'feature_importance': self.feature_importance_df
        }
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        return self.feature_importance_df
    
    def get_prediction_confidence(self, bhk, sqft, bathrooms, location='Unknown', area_type='Unknown'):
        """Get prediction with confidence interval"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get predictions from all trees
        if self.use_advanced_features:
            # Create advanced features for prediction
            bath_per_bhk = bathrooms / bhk
            sqft_per_bhk = sqft / bhk
            
            try:
                location_encoded = self.location_encoder.transform([location])[0]
            except ValueError:
                location_encoded = 0
            
            try:
                area_type_encoded = self.area_encoder.transform([area_type])[0]
            except ValueError:
                area_type_encoded = 0
            
            location_price_mean = 75
            location_price_median = 65
            
            features = [
                bhk, sqft, bathrooms, 0,
                bath_per_bhk, sqft_per_bhk,
                location_encoded, area_type_encoded,
                location_price_mean, location_price_median
            ]
            
            features_scaled = self.scaler.transform([features])
            tree_predictions = [tree.predict(features_scaled)[0] for tree in self.model.estimators_]
        else:
            tree_predictions = [tree.predict([[bhk, sqft, bathrooms]])[0] 
                             for tree in self.model.estimators_]
        
        mean_pred = np.mean(tree_predictions)
        std_pred = np.std(tree_predictions)
        
        return {
            'prediction': mean_pred,
            'confidence_lower': mean_pred - 1.96 * std_pred,
            'confidence_upper': mean_pred + 1.96 * std_pred,
            'std_dev': std_pred
        }
    
    def analyze_feature_interactions(self):
        """Analyze which features work together"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # This is a simplified interaction analysis
        # In practice, you might want to use more sophisticated methods
        importance_df = self.get_feature_importance()
        
        print("\nFeature Importance Analysis:")
        print("="*50)
        
        for idx, row in importance_df.head(10).iterrows():
            feature = row['feature']
            importance = row['importance']
            percentage = (importance / importance_df['importance'].sum()) * 100
            
            print(f"{feature:25}: {importance:.4f} ({percentage:.1f}%)")
        
        return importance_df
