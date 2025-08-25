# sample_data.py

# Optional: a simple list the model can consume via predict_multiple
def get_sample_feature_list():
    # Each tuple is (BHK, Sqft, Bathrooms)
    return [
        (2, 1056, 2),   # e.g., 2 BHK, ~1056 sqft, 2 baths
        (3, 1500, 3),
        (1, 600, 1),
        (4, 2100, 4),
        (2, 900, 2),
        (3, 1800, 3),
    ]

# Optional: a pandas DataFrame with the exact feature column names
def get_sample_dataframe():
    import pandas as pd
    rows = [
        {"BHK": 2, "Sqft": 1056, "No of Bathrooms": 2},
        {"BHK": 3, "Sqft": 1500, "No of Bathrooms": 3},
        {"BHK": 1, "Sqft": 600,  "No of Bathrooms": 1},
        {"BHK": 4, "Sqft": 2100, "No of Bathrooms": 4},
        {"BHK": 2, "Sqft": 900,  "No of Bathrooms": 2},
        {"BHK": 3, "Sqft": 1800, "No of Bathrooms": 3},
    ]
    return pd.DataFrame(rows, columns=["BHK", "Sqft", "No of Bathrooms"])

