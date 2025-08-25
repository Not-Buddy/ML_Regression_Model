"""
Main driver code for Bengaluru House Price Prediction System

Author: Python ML Project
Description: User interface with separate ML implementations
"""

from linear_regression import LinearRegressionPredictor
from random_forest import RandomForestPredictor


def print_header():
    """Print application header"""
    print("="*60)
    print("BENGALURU HOUSE PRICE PREDICTION SYSTEM")
    print("="*60)


def print_dataset_statistics(X, y):
    """Print dataset statistics"""
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(X)}")
    
    # Handle both basic and advanced features
    if 'BHK' in X.columns:
        print(f"Average BHK: {X['BHK'].mean():.2f}")
        print(f"Average Sqft: {X['Sqft'].mean():.2f}")
        if 'No of Bathrooms' in X.columns:
            print(f"Average Bathrooms: {X['No of Bathrooms'].mean():.2f}")
    else:
        print(f"Features: {list(X.columns)}")
    
    print(f"Average Price: {y.mean():.2f} Lakhs")


def handle_custom_prediction(predictor):
    """Handle custom user input for prediction"""
    if predictor.use_advanced_features:
        user_input = input("Enter BHK, Sqft, Bathrooms, Location (optional), Area_Type (optional) - comma-separated: ").strip()
    else:
        user_input = input("Enter BHK, Sqft, Bathrooms (comma-separated): ").strip()
    
    if user_input.lower() == "quit":
        return False
    
    try:
        values = [x.strip() for x in user_input.split(",")]
        
        # Parse basic features
        bhk = float(values[0])
        sqft = float(values[1])
        bathrooms = float(values[2])
        
        # Parse optional features
        location = values[3] if len(values) > 3 and values[3] else 'Unknown'
        area_type = values[4] if len(values) > 4 and values[4] else 'Unknown'
        
        # Make prediction
        predicted_price = predictor.predict_price(bhk, sqft, bathrooms, location, area_type)
        
        print(f"\nInput: BHK={bhk}, Sqft={sqft}, Bathrooms={bathrooms}")
        if predictor.use_advanced_features:
            print(f"Location: {location}, Area Type: {area_type}")
        
        print(f"Predicted Price: ₹{predicted_price:.2f} Lakhs")
        print(f"Predicted Price: ₹{predicted_price*100000:.0f}")
        
        # Show confidence interval for Random Forest
        if isinstance(predictor, RandomForestPredictor):
            try:
                confidence = predictor.get_prediction_confidence(bhk, sqft, bathrooms, location, area_type)
                print(f"Confidence Interval: ₹{confidence['confidence_lower']:.2f} - ₹{confidence['confidence_upper']:.2f} Lakhs")
            except:
                pass
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please enter valid values.")
    except IndexError:
        print("Please enter at least BHK, Sqft, and Bathrooms.")
    
    return True


def handle_demo_predictions(predictor, df_clean):
    """Handle demo predictions from cleaned data"""
    print("\nPredicting on 5 random rows from cleaned data (demo)...")
    demo_df = df_clean.sample(5, random_state=42)
    
    predictions = []
    for _, row in demo_df.iterrows():
        bhk = int(row["BHK"])
        sqft = float(row["Sqft"])
        baths = float(row["No of Bathrooms"])
        location = row.get("Location", "Unknown")
        area_type = row.get("Area_Type", "Unknown")
        
        pred = predictor.predict_price(bhk, sqft, baths, location, area_type)
        predictions.append(pred)
    
    for (_, row), pred in zip(demo_df.iterrows(), predictions):
        bhk = int(row["BHK"])
        sqft = float(row["Sqft"])
        baths = float(row["No of Bathrooms"])
        actual = float(row["Price (In Lakhs)"])
        
        print(f"BHK={bhk}, Sqft={sqft:.0f}, Baths={baths:.0f} -> "
              f"Pred ₹{pred:.2f}L (₹{pred*100000:.0f}), Actual ₹{actual:.2f}L")


def handle_feature_analysis(predictor):
    """Handle feature importance analysis"""
    if not predictor.is_trained:
        print("Model must be trained first!")
        return
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    if isinstance(predictor, RandomForestPredictor):
        predictor.analyze_feature_interactions()
    elif isinstance(predictor, LinearRegressionPredictor):
        importance_df = predictor.get_feature_importance()
        print("\nFeature Coefficients (Linear Regression):")
        print(importance_df.head(10))
        
        print(f"\nLinear Equation:")
        print(predictor.get_model_equation())


def display_menu():
    """Display the main menu"""
    return (
        "\nChoose an option:\n"
        "[C]ustom input (enter house details)\n"
        "[D]emo from cleaned data (random 5 rows)\n"
        "[F]eature analysis (importance & coefficients)\n"
        "[Q]uit\n"
        "Your choice: "
    )


def run_interactive_session(predictor, df_clean):
    """Run the interactive prediction session"""
    print("\n" + "="*60)
    print("INTERACTIVE HOUSE PRICE PREDICTION")
    print("="*60)
    
    print(f"🤖 Model: {predictor.model_type.title()}")
    if predictor.use_advanced_features:
        print("ℹ️  Advanced features enabled - location and area type will improve accuracy!")
    
    menu = display_menu()
    
    while True:
        try:
            choice = input(menu).strip().lower()
            
            if choice in ("q", "quit"):
                print("Thank you for using the House Price Prediction System!")
                break
            
            elif choice in ("c", "custom"):
                if not handle_custom_prediction(predictor):
                    break
            
            elif choice in ("d", "demo"):
                handle_demo_predictions(predictor, df_clean)
            
            elif choice in ("f", "feature"):
                handle_feature_analysis(predictor)
            
            else:
                print("Invalid option. Please choose C, D, F, or Q.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


def main():
    """Main function to run the house price prediction"""
    print_header()
    
    # Ask user for model configuration
    print("\nChoose model type:")
    print("[1] Linear Regression (simple, interpretable)")
    print("[2] Random Forest (advanced, higher accuracy)")
    
    model_choice = input("Enter choice (1 or 2): ").strip()
    
    print("\nChoose feature set:")
    print("[1] Basic features (BHK, Sqft, Bathrooms)")
    print("[2] Advanced features (includes location, area type, etc.)")
    
    feature_choice = input("Enter choice (1 or 2): ").strip()
    
    use_advanced_features = feature_choice == "2"
    
    # Initialize the appropriate predictor
    if model_choice == "1":
        predictor = LinearRegressionPredictor(use_advanced_features=use_advanced_features)
        print("📊 Linear Regression model selected!")
    else:
        predictor = RandomForestPredictor(use_advanced_features=use_advanced_features)
        print("🌲 Random Forest model selected!")
    
    if use_advanced_features:
        print("🚀 Advanced feature engineering enabled!")
    else:
        print("📈 Basic features enabled!")
    
    # Configuration
    CSV_FILE = "../data/Bengaluru_House_Data_ML_Project.csv"
    
    try:
        # Load and prepare data
        print("\nLoading and cleaning data...")
        df_clean = predictor.load_and_clean_data(CSV_FILE)
        
        # Prepare features
        X, y = predictor.prepare_features(df_clean)
        print_dataset_statistics(X, y)
        
        # Train model
        print(f"\nTraining the {predictor.model_type} model...")
        metrics = predictor.train_model(X, y)
        
        # Run interactive session
        run_interactive_session(predictor, df_clean)
        
    except FileNotFoundError:
        print(f"Error: Could not find the data file '{CSV_FILE}'")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
