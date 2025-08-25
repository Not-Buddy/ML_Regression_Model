# Bengaluru House Price Prediction

A machine learning project that predicts house prices in Bengaluru using Linear Regression based on house features like BHK, Square Feet, and Number of Bathrooms.

## Features

- **Data Cleaning**: Handles various data formats and removes outliers
- **Linear Regression Model**: Uses scikit-learn for price prediction
- **Interactive Interface**: Command-line interface for real-time predictions
- **Feature Analysis**: Shows model coefficients and performance metrics

## Dataset

The project uses the Bengaluru House Data dataset with the following key features:
- **BHK**: Number of bedrooms (1-10)
- **Square Feet**: Total area of the house (300-30,000 sqft)
- **Bathrooms**: Number of bathrooms (1-15)
- **Price**: House price in Lakhs (target variable)

## Installation

1. Clone or download the project files
2. Install required dependencies:

```bash
python3 -m venv myenv
source myenv/bin/activate

```

```bash
pip install -r requirements.txt
```

when you are done use to deactivate virtual environment
```bash
deactivate
```
## Usage

### Running the Interactive System

```bash
python3 house_price_predictor.py
```

### Using the Predictor Class

```python
from house_price_predictor import HousePricePredictor

# Initialize predictor
predictor = HousePricePredictor()

# Load and train model
df_clean = predictor.load_and_clean_data('Bengaluru_House_Data-ML-Project.csv')
X, y = predictor.prepare_features(df_clean)
metrics = predictor.train_model(X, y)

# Make predictions
price = predictor.predict_price(bhk=3, sqft=1500, bathrooms=2)
print(f"Predicted Price: ₹{price:.2f} Lakhs")
```

## Model Performance

- **R² Score**: ~0.54 (explains 54% of price variance)
- **RMSE**: ~56 Lakhs
- **Features Impact**:
  - Bathrooms: Highest impact on price
  - Square Feet: Moderate positive correlation
  - BHK: Lower but positive impact

## Project Structure

```
├── house_price_predictor.py    # Main prediction system
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
└── Bengaluru_House_Data-ML-Project.csv    # Dataset
```

## Example Predictions

| BHK | Sqft | Bath | Predicted Price |
|-----|------|------|----------------|
| 2   | 1200 | 2    | ₹68.88 Lakhs   |
| 3   | 1500 | 3    | ₹106.75 Lakhs  |
| 1   | 800  | 1    | ₹26.41 Lakhs   |
| 4   | 2500 | 4    | ₹176.78 Lakhs  |

## Future Enhancements

- Add more features (location, age, amenities)
- Implement other ML algorithms (Random Forest, XGBoost)
- Create a web interface using Flask/Django
- Add data visualization and analysis features

## License

This project is for educational purposes.

