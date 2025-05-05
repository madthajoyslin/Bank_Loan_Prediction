import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import traceback # Keep for better error logging
import random # Import the random module

# Initialize Flask app
app = Flask(__name__)

# --- Load Saved Artifacts ---
try:
    # Load the scaler
    scaler_filename = 'standard_scaler.pkl'
    scaler = joblib.load(scaler_filename)
    print(f"Scaler loaded from {scaler_filename}")

    # Load the trained model
    model_filename = 'best_xgboost_model.pkl'
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {model_filename}")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Ensure 'standard_scaler.pkl' and 'best_xgboost_model.pkl' are in the same directory as app.py")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit()

# --- Define Expected Feature Order ---
# This remains the same as the model expects these features after preprocessing
expected_feature_names = [
    'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'credit_score', 'previous_loan_defaults_on_file', # Still needed by model
    'person_gender_female', 'person_gender_male',
    'person_education_Associate', 'person_education_Bachelor',
    'person_education_Doctorate', 'person_education_High School', 'person_education_Master',
    'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER',
    'person_home_ownership_OWN', 'person_home_ownership_RENT',
    'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
    'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL',
    'loan_intent_VENTURE'
]

# --- Define columns to be one-hot encoded (used in preprocessing) ---
categorical_cols = ["person_gender", "person_education", "person_home_ownership", "loan_intent"]

# --- Preprocessing Function (Handles the derived 'previous_loan_defaults_on_file') ---
def preprocess_input(data_dict):
    """
    Takes a dictionary of raw input data (including the DERIVED/GENERATED values)
    and preprocesses it to match the format used for model training.
    """
    try:
        # 1. Convert to DataFrame
        # Check if essential derived/generated values are present before creating DataFrame
        required_keys = ['credit_score', 'previous_loan_defaults_on_file',
                         'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
        missing_keys = [key for key in required_keys if key not in data_dict]
        if missing_keys:
             raise ValueError(f"Missing required derived/generated keys in data passed to preprocess_input: {missing_keys}")

        df_input = pd.DataFrame([data_dict])

        # 2. Map 'previous_loan_defaults_on_file' (Value should be "Yes" or "No" at this point)
        # This mapping is crucial as the model expects 0 or 1
        df_input['previous_loan_defaults_on_file'] = df_input['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})
        # Ensure mapping worked, default to 0 if somehow it was neither Yes/No
        df_input['previous_loan_defaults_on_file'] = df_input['previous_loan_defaults_on_file'].fillna(0).astype(int)


        # 3. Convert numerical fields explicitly
        # Note: credit_score, previous_loan_defaults_on_file, and the GENERATED/CALCULATED fields are included
        numeric_fields = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                          'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                          'credit_score', 'previous_loan_defaults_on_file']
        for col in numeric_fields:
             # Use errors='coerce' to handle potential issues if non-numeric data slips through
             # This will also handle the generated float values correctly
             df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

        # Check for NaN values AFTER numeric conversion
        if df_input[numeric_fields].isnull().any().any():
             nan_cols = df_input[numeric_fields].isnull().any()
             print(f"NaN values detected after numeric conversion in columns: {nan_cols[nan_cols].index.tolist()}")
             # Also print the input dict again to see what caused NaN
             print("Input Dictionary causing NaN:", data_dict)
             raise ValueError("Invalid numeric input detected (resulted in NaN). Check values like Credit Score or generated numbers.")


        # 4. Apply One-Hot Encoding for original categorical columns
        df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=False)

        # 5. Align Columns with Training Data
        # Make sure df_aligned contains all columns needed by the scaler/model
        df_aligned = df_encoded.reindex(columns=expected_feature_names, fill_value=0)

        # Check alignment - Ensure all expected columns are present
        if df_aligned.isnull().any().any():
            print("Warning: Null values detected after reindexing before scaling.")
            print(df_aligned.isnull().sum())
            # Consider adding more debugging here if this warning occurs

        # 6. Scale the data using the loaded scaler
        # Ensure the scaler is fitted on the same columns in the same order
        scaled_features = scaler.transform(df_aligned[expected_feature_names]) # Explicitly use expected_feature_names order

        return scaled_features

    except KeyError as e:
        print(f"Preprocessing Error: Missing key {e}")
        # Provide a more specific message if possible
        raise ValueError(f"Missing input field needed for preprocessing: {e}")
    except ValueError as e:
        print(f"Preprocessing Error: {e}")
        # More specific error message might be helpful
        raise ValueError(f"Invalid input value or preprocessing issue: {e}")
    except Exception as e:
        print(f"Unexpected Preprocessing Error: {e}")
        print(traceback.format_exc()) # Log traceback for unexpected errors
        raise RuntimeError(f"An unexpected error occurred during preprocessing: {e}")


# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the main HTML form."""
    return render_template('index.html') # Pass no prediction initially

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, derives/generates missing fields, preprocesses, predicts, and returns result."""
    prediction_text = None # Initialize to None
    try:
        # 1. Get data from form (excluding the fields removed from HTML)
        form_data = request.form.to_dict()
        print("Received form data:", form_data) # Debugging

        # --- 2. VALIDATE and GET required inputs from form ---
        loan_intent = form_data.get('loan_intent')
        credit_score_str = form_data.get('credit_score')

        if not loan_intent:
            raise ValueError("Loan Intent is missing.")
        if credit_score_str is None or credit_score_str.strip() == "":
            raise ValueError("Credit Score is required.")

        # Validate and convert credit score
        try:
            credit_score = float(credit_score_str) # Convert to float for comparison
        except ValueError:
            raise ValueError("Invalid Credit Score entered. Please provide a number.")

        # --- 3. DERIVE/CALCULATE

     # a) Calculate 'cb_person_cred_hist_length' based on 'loan_intent'
        if loan_intent == 'HOMEIMPROVEMENT':
            cb_hist_length = 10
        else:
            cb_hist_length = 5
        form_data['cb_person_cred_hist_length'] = cb_hist_length
        print(f"Calculated Credit History Length: {cb_hist_length} based on Intent: {loan_intent}")

    # b) Generate 'loan_int_rate' (based on dataset sample: ~7 to 17)
        # Using random.uniform for float values within a range
        generated_int_rate = round(random.uniform(7.0, 17.0), 2)
        form_data['loan_int_rate'] = generated_int_rate
        print(f"Generated Loan Interest Rate: {generated_int_rate}")

     # c) Generate 'loan_percent_income' (based on dataset sample: ~0.05 to 0.60)
        generated_percent_income = round(random.uniform(0.05, 0.60), 2)
        form_data['loan_percent_income'] = generated_percent_income
        print(f"Generated Loan Percent Income: {generated_percent_income}")

        # d) DERIVE 'previous_loan_defaults_on_file' from 'credit_score' (logic remains same)
        if credit_score < 550:
            derived_defaults_status = "Yes"
        else:
            derived_defaults_status = "No"
        form_data['previous_loan_defaults_on_file'] = derived_defaults_status
        print(f"Derived Default Status: {derived_defaults_status} based on Score: {credit_score}")

        # --- 4. Log the complete data going into preprocessing ---
        print(f"Data prepared for preprocessing (incl. derived/generated fields): {form_data}") # Debugging

        # --- 5. Preprocess the input data ---
        # The preprocess_input function now expects the derived/generated values
        processed_data = preprocess_input(form_data)
        print("Data shape after preprocessing and scaling:", processed_data.shape) # Debugging

        # --- 6. Make prediction ---
        prediction = model.predict(processed_data)
        output_value = int(prediction[0]) # Get the single prediction value (0 or 1)

        # --- 7. Interpret prediction ---
        if output_value == 0: # Assuming 0 means DEFAULT/Not Approved
            prediction_text = "Thank you for your application. Based on the information provided, you are not eligible for the loan"
        else: # Assuming 1 means Repaid/Approved
            prediction_text = "Congratulations! You are eligible for the loan."

        print(f"Prediction: {output_value} -> {prediction_text}")

    except (ValueError, RuntimeError) as e:
        # Handle specific errors (missing input, bad conversion, preprocessing issues)
        prediction_text = f"Error during prediction: {e}. Please check your input."
        print(traceback.format_exc()) # Log traceback for debugging
    except Exception as e:
        # Handle unexpected model or other server errors
        prediction_text = f"An unexpected server error occurred: {e}"
        print(f"Prediction Error: {e}") # Log the full error
        print(traceback.format_exc())

    # Render the template again, passing the prediction text and optional debug info
    return render_template('index.html', prediction_text=prediction_text)

# --- Run the App ---
if __name__ == "__main__":
    # Important: Set debug=False for production deployment
    app.run(debug=True)