"""
Data cleaning and preprocessing utilities for House Price Prediction

Author: Python ML Project
Description: Data cleaning, feature engineering, and preprocessing functions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

def extract_bhk(bedrooms_str):
    """Extract BHK number from bedroom string"""
    if pd.isna(bedrooms_str):
        return np.nan
    
    bedrooms_str = str(bedrooms_str).strip()
    
    if 'BHK' in bedrooms_str:
        return int(bedrooms_str.split(' ')[0])
    elif 'Bedroom' in bedrooms_str:
        return int(bedrooms_str.split(' ')[0])
    elif 'RK' in bedrooms_str:
        return 1
    else:
        try:
            return int(bedrooms_str)
        except ValueError:
            return np.nan

def clean_sqft(sqft_str):
    """Clean and convert square feet data"""
    if pd.isna(sqft_str):
        return np.nan
    
    sqft_str = str(sqft_str).strip()
    
    if ' - ' in sqft_str:
        try:
            parts = sqft_str.split(' - ')
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return np.nan
    
    # Clean special units
    sqft_str = sqft_str.replace('Sq. Meter', '').replace('Sq. Yards', '').replace('Acres', '').replace('Guntha', '').replace('Perch', '')
    
    try:
        numbers = re.findall(r'\d+\.?\d*', sqft_str)
        if numbers:
            return float(numbers[0])
        else:
            return np.nan
    except:
        return np.nan

def create_advanced_features(df, location_encoder, area_encoder):
    """Create more predictive features"""
    print("Creating advanced features...")
    
    # Basic derived features
    df['bath_per_bhk'] = df['No of Bathrooms'] / df['BHK']
    df['sqft_per_bhk'] = df['Sqft'] / df['BHK']
    
    # Handle Balcony column if it exists
    if 'Balcony' in df.columns:
        df['Balcony'] = df['Balcony'].fillna(0)
    else:
        df['Balcony'] = 0
        
    # Location encoding (crucial for accuracy)
    df['Location'] = df['Location'].fillna('Unknown')
    df['location_encoded'] = location_encoder.fit_transform(df['Location'])
    
    # Area type encoding
    df['Area_Type'] = df['Area_Type'].fillna('Unknown')
    df['area_type_encoded'] = area_encoder.fit_transform(df['Area_Type'])
    
    # Location-based price statistics (very powerful feature)
    location_stats = df.groupby('Location')['Price (In Lakhs)'].agg(['mean', 'median']).fillna(0)
    location_stats.columns = ['location_price_mean', 'location_price_median']
    df = df.merge(location_stats, left_on='Location', right_index=True, how='left')
    
    # Price per sqft (calculated during training, not used as feature)
    df['price_per_sqft'] = df['Price (In Lakhs)'] * 100000 / df['Sqft']
    
    print(f"Advanced features created successfully!")
    return df

def remove_outliers_advanced(df):
    """Advanced outlier removal using price per sqft"""
    print("Removing outliers using advanced method...")
    
    # Calculate price per sqft
    df['price_per_sqft'] = df['Price (In Lakhs)'] * 100000 / df['Sqft']
    
    # Remove outliers by location groups
    def remove_location_outliers(group):
        if len(group) < 5:  # Keep small groups as-is
            return group
        Q1 = group['price_per_sqft'].quantile(0.25)
        Q3 = group['price_per_sqft'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return group[(group['price_per_sqft'] >= lower_bound) & 
                    (group['price_per_sqft'] <= upper_bound)]
    
    df_clean = df.groupby('Location').apply(remove_location_outliers).reset_index(drop=True)
    
    # Additional business logic filters
    df_clean = df_clean[
        (df_clean['Sqft'] >= 300) & (df_clean['Sqft'] <= 30000) &
        (df_clean['BHK'] >= 1) & (df_clean['BHK'] <= 10) &
        (df_clean['No of Bathrooms'] >= 1) & (df_clean['No of Bathrooms'] <= df_clean['BHK'] + 2) &
        (df_clean['Price (In Lakhs)'] >= 10) & (df_clean['Price (In Lakhs)'] <= 1000)
    ]
    
    print(f"Outliers removed. Rows: {len(df)} -> {len(df_clean)}")
    return df_clean

def load_and_clean_data(csv_file_path, use_advanced_features=True, location_encoder=None, area_encoder=None):
    """Load and clean the dataset with advanced processing"""
    df = pd.read_csv(csv_file_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Extract BHK and clean square feet
    df['BHK'] = df['No of Bedrooms'].apply(extract_bhk)
    df['Sqft'] = df['Total_Sqft'].apply(clean_sqft)
    
    # Remove rows with missing essential data
    df_clean = df.dropna(subset=['BHK', 'Sqft', 'No of Bathrooms', 'Price (In Lakhs)'])
    
    if use_advanced_features:
        # Use advanced outlier removal
        df_clean = remove_outliers_advanced(df_clean)
        # Create advanced features
        if location_encoder is not None and area_encoder is not None:
            df_clean = create_advanced_features(df_clean, location_encoder, area_encoder)
    else:
        # Basic outlier removal
        df_clean = df_clean[
            (df_clean['Sqft'] >= 300) & (df_clean['Sqft'] <= 30000) &
            (df_clean['BHK'] >= 1) & (df_clean['BHK'] <= 10) &
            (df_clean['No of Bathrooms'] >= 1) & (df_clean['No of Bathrooms'] <= 15) &
            (df_clean['Price (In Lakhs)'] >= 5) & (df_clean['Price (In Lakhs)'] <= 500)
        ]
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Data removed: {df.shape[0] - df_clean.shape[0]} rows")
    
    return df_clean

def prepare_features_basic(df):
    """Prepare basic features for training"""
    X = df[['BHK', 'Sqft', 'No of Bathrooms']].copy()
    y = df['Price (In Lakhs)'].copy()
    feature_columns = X.columns.tolist()
    return X, y, feature_columns

def prepare_features_advanced(df, scaler):
    """Enhanced feature preparation with scaling"""
    print("Preparing advanced features...")
    
    # Define feature columns
    feature_columns = [
        'BHK', 'Sqft', 'No of Bathrooms', 'Balcony',
        'bath_per_bhk', 'sqft_per_bhk', 
        'location_encoded', 'area_type_encoded',
        'location_price_mean', 'location_price_median'
    ]
    
    # Handle missing values
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    X = df[feature_columns].copy()
    y = df['Price (In Lakhs)'].copy()
    
    # Feature scaling for better model performance
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
    
    print(f"Features prepared: {X_scaled.shape}")
    return X_scaled, y, feature_columns

def prepare_features(df, use_advanced_features=True, scaler=None):
    """Prepare features for training (basic or advanced)"""
    if use_advanced_features:
        if scaler is None:
            raise ValueError("Scaler is required for advanced features")
        return prepare_features_advanced(df, scaler)
    else:
        return prepare_features_basic(df)
