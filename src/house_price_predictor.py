
"""
Bengaluru House Price Prediction using Linear Regression
Author: Python ML Project
Description: A machine learning model to predict house prices in Bengaluru 
based on BHK, Square Feet, and Number of Bathrooms.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sample_data import get_sample_feature_list
import re
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    """
    A class to predict house prices using Linear Regression
    """

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    def extract_bhk(self, bedrooms_str):
        """Extract BHK number from bedroom string"""
        if pd.isna(bedrooms_str):
            return np.nan

        bedrooms_str = str(bedrooms_str).strip()

        # Handle BHK format (e.g., "2 BHK", "3 BHK")
        if 'BHK' in bedrooms_str:
            return int(bedrooms_str.split(' ')[0])

        # Handle Bedroom format (e.g., "4 Bedroom", "2 Bedroom")
        elif 'Bedroom' in bedrooms_str:
            return int(bedrooms_str.split(' ')[0])

        # Handle RK format (e.g., "1 RK")
        elif 'RK' in bedrooms_str:
            return 1

        # Try to extract number directly
        else:
            try:
                return int(bedrooms_str)
            except ValueError:
                return np.nan

    def clean_sqft(self, sqft_str):
        """Clean and convert square feet data"""
        if pd.isna(sqft_str):
            return np.nan

        sqft_str = str(sqft_str).strip()

        # Handle range format (e.g., "1200 - 1400")
        if ' - ' in sqft_str:
            try:
                parts = sqft_str.split(' - ')
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                return np.nan

        # Handle special units and clean the string
        sqft_str = sqft_str.replace('Sq. Meter', '').replace('Sq. Yards', '').replace('Acres', '').replace('Guntha', '').replace('Perch', '')

        try:
            # Extract numeric value using regex
            numbers = re.findall(r'\d+\.?\d*', sqft_str)
            if numbers:
                return float(numbers[0])
            else:
                return np.nan
        except:
            return np.nan

    def load_and_clean_data(self, csv_file_path):
        """Load and clean the dataset"""
        df = pd.read_csv(csv_file_path)

        print(f"Original dataset shape: {df.shape}")

        # Extract BHK and clean square feet
        df['BHK'] = df['No of Bedrooms'].apply(self.extract_bhk)
        df['Sqft'] = df['Total_Sqft'].apply(self.clean_sqft)

        # Remove rows with missing essential data
        df_clean = df.dropna(subset=['BHK', 'Sqft', 'No of Bathrooms', 'Price (In Lakhs)'])

        # Remove outliers
        df_clean = df_clean[
            (df_clean['Sqft'] >= 300) & (df_clean['Sqft'] <= 30000) &
            (df_clean['BHK'] >= 1) & (df_clean['BHK'] <= 10) &
            (df_clean['No of Bathrooms'] >= 1) & (df_clean['No of Bathrooms'] <= 15) &
            (df_clean['Price (In Lakhs)'] >= 5) & (df_clean['Price (In Lakhs)'] <= 500)
        ]

        print(f"Cleaned dataset shape: {df_clean.shape}")
        print(f"Data removed: {df.shape[0] - df_clean.shape[0]} rows")

        return df_clean

    def prepare_features(self, df):
        """Prepare features for training"""
        X = df[['BHK', 'Sqft', 'No of Bathrooms']].copy()
        y = df['Price (In Lakhs)'].copy()

        return X, y

    def train_model(self, X, y):
        """Train the linear regression model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate the model
        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")

        # Show feature importance (coefficients)
        feature_names = ['BHK', 'Sqft', 'Bathrooms']
        coefficients = self.model.coef_
        print(f"\nModel Coefficients:")
        for name, coef in zip(feature_names, coefficients):
            print(f"{name}: {coef:.4f}")
        print(f"Intercept: {self.model.intercept_:.4f}")

        return {
            'r2_score': r2,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'coefficients': dict(zip(feature_names, coefficients)),
            'intercept': self.model.intercept_
        }

    def predict_price(self, bhk, sqft, bathrooms):
        """Predict house price for given features"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please call train_model() first.")

        # Validate inputs
        if bhk <= 0 or sqft <= 0 or bathrooms <= 0:
            raise ValueError("All features must be positive values")

        prediction = self.model.predict([[bhk, sqft, bathrooms]])
        return prediction[0]

    def predict_multiple(self, feature_list):
        """Predict prices for multiple houses"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please call train_model() first.")

        predictions = []
        for bhk, sqft, bathrooms in feature_list:
            price = self.predict_price(bhk, sqft, bathrooms)
            predictions.append(price)

        return predictions

def main():
    """Main function to run the house price prediction"""

    print("="*60)
    print("BENGALURU HOUSE PRICE PREDICTION SYSTEM")
    print("="*60)

    # Initialize predictor
    predictor = HousePricePredictor()

    # Point to the actual CSV on disk (adjust path as needed)
    # Matches the provided attachment name
    CSV_FILE = "../data/Bengaluru_House_Data_ML_Project.csv"

    # Load and prepare data
    print("\nLoading and cleaning data...")
    df_clean = predictor.load_and_clean_data(CSV_FILE)

    # Prepare features
    X, y = predictor.prepare_features(df_clean)

    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Average BHK: {X['BHK'].mean():.2f}")
    print(f"Average Sqft: {X['Sqft'].mean():.2f}")
    print(f"Average Bathrooms: {X['No of Bathrooms'].mean():.2f}")
    print(f"Average Price: {y.mean():.2f} Lakhs")

    # Train model
    print("\nTraining the Linear Regression model...")
    metrics = predictor.train_model(X, y)

    # Built-in sample feature tuples: (BHK, Sqft, Bathrooms)
    sample_features = [
        (2, 1100, 2),
        (3, 1600, 3),
        (1, 600, 1),
        (4, 2200, 4),
        (2, 900, 2),
    ]

    # Interactive prediction system
    print("\n" + "="*60)
    print("INTERACTIVE HOUSE PRICE PREDICTION")
    print("="*60)

    menu = (
        "\nChoose an option:\n"
        "[C]ustom input (enter one set of BHK, Sqft, Bathrooms)\n"
        "[S]ample predictions (built-in examples)\n"
        "[D]emo from cleaned data (random 5 rows)\n"
        "[Q]uit\n"
        "Your choice: "
    )

    while True:
        try:
            choice = input(menu).strip().lower()

            if choice in ("q", "quit"):
                print("Thank you for using the House Price Prediction System!")
                break

            elif choice in ("c", "custom"):
                # Custom single input (keeps your original flow)
                user_input = input("Enter BHK, Sqft, Bathrooms (comma-separated): ").strip()
                if user_input.lower() == "quit":
                    print("Thank you for using the House Price Prediction System!")
                    break

                values = [float(x.strip()) for x in user_input.split(",")]
                if len(values) != 3:
                    print("Please enter exactly 3 values: BHK, Sqft, Bathrooms")
                    continue

                bhk, sqft, bathrooms = values
                predicted_price = predictor.predict_price(bhk, sqft, bathrooms)
                print(f"\nPredicted Price: ₹{predicted_price:.2f} Lakhs")
                print(f"Predicted Price: ₹{predicted_price*100000:.0f}")

            elif choice in ("s", "sample"):
                # Run on built-in sample list
                print("\nRunning sample predictions...")
                preds = predictor.predict_multiple(sample_features)
                for (bhk, sqft, baths), price in zip(sample_features, preds):
                    print(f"BHK={bhk}, Sqft={sqft}, Baths={baths} -> ₹{price:.2f} Lakhs | ₹{price*100000:.0f}")

            elif choice in ("d", "demo"):
                # Predict on a few rows from the cleaned data (random 5 rows)
                print("\nPredicting on 5 random rows from cleaned data (demo)...")
                demo_df = df_clean.sample(5, random_state=42)
                feature_list = list(zip(demo_df["BHK"], demo_df["Sqft"], demo_df["No of Bathrooms"]))
                demo_preds = predictor.predict_multiple(feature_list)

                for (_, row), pred in zip(demo_df.iterrows(), demo_preds):
                    bhk = int(row["BHK"])
                    sqft = float(row["Sqft"])
                    baths = float(row["No of Bathrooms"])
                    actual = float(row["Price (In Lakhs)"])
                    print(
                        f"BHK={bhk}, Sqft={sqft:.0f}, Baths={baths:.0f} -> "
                        f"Pred ₹{pred:.2f}L (₹{pred*100000:.0f}), Actual ₹{actual:.2f}L"
                    )

            else:
                print("Invalid option. Please choose C, S, D, or Q.")

        except ValueError as e:
            print(f"Error: {e}")
            print("Please enter valid numeric values.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

