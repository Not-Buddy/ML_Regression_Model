"""
Linear Regression implementation for House Price Prediction

Author: Python ML Project
Description: Linear Regression specific implementation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from house_price_predictor import BaseHousePricePredictor

class LinearRegressionPredictor(BaseHousePricePredictor):
    """
    Linear Regression implementation for house price prediction
    """
    
    def __init__(self, use_advanced_features=False):
        super().__init__(use_advanced_features)
        self.model = LinearRegression()
        self.model_type = 'linear'
    
    def train_model(self, X, y):
        """Train the Linear Regression model"""
        print("Training Linear Regression model...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
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
        
        print(f"\nLinear Regression Performance:")
        print(f"Test Set R² Score: {r2:.4f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")
        
        # Show coefficients
        print(f"\nModel Coefficients:")
        for name, coef in zip(self.feature_columns, self.model.coef_):
            print(f"{name}: {coef:.4f}")
        print(f"Intercept: {self.model.intercept_:.4f}")
        
        # Feature importance based on absolute coefficient values
        if self.use_advanced_features:
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'coefficient': self.model.coef_,
                'abs_coefficient': np.abs(self.model.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            print(f"\nTop 5 Most Important Features (by coefficient magnitude):")
            print(feature_importance.head()[['feature', 'coefficient']])
        
        return {
            'r2_score': r2,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'coefficients': dict(zip(self.feature_columns, self.model.coef_)),
            'intercept': self.model.intercept_
        }
    
    def get_feature_importance(self):
        """Get feature importance based on coefficient values"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def get_model_equation(self):
        """Get the linear equation as a string"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        equation_parts = [f"{self.model.intercept_:.4f}"]
        
        for feature, coef in zip(self.feature_columns, self.model.coef_):
            if coef >= 0:
                equation_parts.append(f" + {coef:.4f} * {feature}")
            else:
                equation_parts.append(f" - {abs(coef):.4f} * {feature}")
        
        return "Price = " + "".join(equation_parts)
