import joblib
import pandas as pd

def test_pipeline():
    print("Testing loaded artifacts...")
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    metrics = joblib.load('metrics.pkl')
    results_df = joblib.load('results_df.pkl')
    
    print(f"Features: {features}")
    print(f"Metrics: {metrics}")

    # Sample input test
    sample_input = {
        'no_of_dependents': 2,
        'education': 0, # Graduate
        'self_employed': 0, # No
        'income_annum': 6000000,
        'loan_amount': 12000000,
        'loan_term': 10,
        'cibil_score': 780,
        'residential_assets_value': 5000000,
        'commercial_assets_value': 2000000,
        'luxury_assets_value': 10000000,
        'bank_asset_value': 4000000
    }

    input_df = pd.DataFrame([sample_input])[features]
    scaled_df = scaler.transform(input_df)
    
    pred = model.predict(scaled_df)[0]
    prob = model.predict_proba(scaled_df)[0, 1]

    print(f"\nSample Input Prediction:")
    print(f"Approved: {'Yes' if pred == 1 else 'No'}")
    print(f"Approval Probability: {prob * 100:.2f}%")
    print("✅ Model prediction pipeline test passed successfully!")

if __name__ == '__main__':
    test_pipeline()
