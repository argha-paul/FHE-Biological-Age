import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import shutil
from pathlib import Path
import os

# Ensure the directory exists
os.makedirs("./plots", exist_ok=True)

# Import Concrete ML
from concrete.ml.sklearn import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer

def load_data(file_path="biomarker_dataset.csv"):
    """Load the biomarker dataset."""
    df = pd.read_csv(file_path)

    # Separate features and targets
    X = df.drop(['chronological_age', 'biological_age', 'aging_pace'], axis=1)
    y_bio_age = df['biological_age']
    y_aging_pace = df['aging_pace']

    # Train-test split
    X_train_bio, X_test_bio, y_train_bio, y_test_bio = train_test_split(
        X, y_bio_age, test_size=0.2, random_state=42
    )

    X_train_pace, X_test_pace, y_train_pace, y_test_pace = train_test_split(
        X, y_aging_pace, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler_bio = StandardScaler()
    X_train_bio_scaled = scaler_bio.fit_transform(X_train_bio)
    X_test_bio_scaled = scaler_bio.transform(X_test_bio)

    scaler_pace = StandardScaler()
    X_train_pace_scaled = scaler_pace.fit_transform(X_train_pace)
    X_test_pace_scaled = scaler_pace.transform(X_test_pace)

    return {
        'bio_age': {
            'X_train': X_train_bio,
            'X_test': X_test_bio,
            'y_train': y_train_bio,
            'y_test': y_test_bio,
            'X_train_scaled': X_train_bio_scaled,
            'X_test_scaled': X_test_bio_scaled,
            'scaler': scaler_bio
        },
        'aging_pace': {
            'X_train': X_train_pace,
            'X_test': X_test_pace,
            'y_train': y_train_pace,
            'y_test': y_test_pace,
            'X_train_scaled': X_train_pace_scaled,
            'X_test_scaled': X_test_pace_scaled,
            'scaler': scaler_pace
        },
        'feature_names': X_train_bio.columns.tolist()
    }

def train_and_convert_bio_age_model(data):
    """Train and convert the biological age model to FHE."""
    print("\n--- Training Biological Age FHE Model ---")

    # Extract data
    X_train = data['bio_age']['X_train_scaled']
    y_train = data['bio_age']['y_train']
    X_test = data['bio_age']['X_test_scaled']
    y_test = data['bio_age']['y_test']

    # For FHE compatibility, we'll try different models
    # Start with Linear Regression (most FHE-friendly)
    print("Training Linear Regression model...")
    fhe_linear = LinearRegression()
    fhe_linear.fit(X_train, y_train)

    # Evaluate linear model
    y_pred_linear = fhe_linear.predict(X_test)
    linear_mae = mean_absolute_error(y_test, y_pred_linear)
    linear_r2 = r2_score(y_test, y_pred_linear)
    print(f"Linear Regression - MAE: {linear_mae:.3f}, R²: {linear_r2:.3f}")

    # Try Decision Tree (pruned for FHE compatibility)
    print("Training Decision Tree model...")
    fhe_dt = DecisionTreeRegressor(max_depth=5)  # Limit depth for FHE
    fhe_dt.fit(X_train, y_train)

    # Evaluate decision tree model
    y_pred_dt = fhe_dt.predict(X_test)
    dt_mae = mean_absolute_error(y_test, y_pred_dt)
    dt_r2 = r2_score(y_test, y_pred_dt)
    print(f"Decision Tree - MAE: {dt_mae:.3f}, R²: {dt_r2:.3f}")

    # Try Random Forest (limited for FHE compatibility)
    print("Training Random Forest model...")
    fhe_rf = RandomForestRegressor(n_estimators=5, max_depth=4)  # Limited for FHE
    fhe_rf.fit(X_train, y_train)

    # Evaluate random forest model
    y_pred_rf = fhe_rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)
    print(f"Random Forest - MAE: {rf_mae:.3f}, R²: {rf_r2:.3f}")

    # Select best model based on MAE
    models = {
        'Linear': (fhe_linear, linear_mae, linear_r2, y_pred_linear),
        'Decision Tree': (fhe_dt, dt_mae, dt_r2, y_pred_dt),
        'Random Forest': (fhe_rf, rf_mae, rf_r2, y_pred_rf)
    }

    best_model_name = min(models, key=lambda k: models[k][1])
    best_model, best_mae, best_r2, y_pred = models[best_model_name]

    print(f"\nBest FHE-compatible model: {best_model_name}")
    print(f"MAE: {best_mae:.3f}, R²: {best_r2:.3f}")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Biological Age')
    plt.ylabel('Predicted Biological Age')
    plt.title(f'FHE Model: Actual vs Predicted - {best_model_name}')
    plt.tight_layout()
    plt.savefig(f"./plots/fhe_bio_age_predictions_{best_model_name.lower().replace(' ', '_')}.png")
    plt.close()

    # Compile the model for FHE
    print("\nCompiling model for FHE...")

    # Use a small subset for compilation to save time
    X_sample = X_train[:100]

    # Configure quantization
    best_model.compile(X_sample)

    # Generate keys (dev only - in production this would be different)
    fhe_circuit = best_model.fhe_circuit

    print("Model compiled for FHE!")

    # Save the FHE model and related components
    # joblib.dump(best_model, "fhe_bio_age_model.joblib")
    # best_model.save("fhe_bio_age_model")
    path_to_model = Path("./dev").resolve()
    if path_to_model.exists():
        shutil.rmtree(path_to_model)
    fhe_model_dev = FHEModelDev(path_to_model,best_model)
    fhe_model_dev.save(via_mlir=True)
    joblib.dump(data['bio_age']['scaler'], "./plots/bio_age_scaler.joblib")

    return best_model, best_model_name

def train_and_convert_aging_pace_model(data):
    """Train and convert the aging pace model to FHE."""
    print("\n--- Training Aging Pace FHE Model ---")

    # Extract data
    X_train = data['aging_pace']['X_train_scaled']
    y_train = data['aging_pace']['y_train']
    X_test = data['aging_pace']['X_test_scaled']
    y_test = data['aging_pace']['y_test']

    # For FHE compatibility, we'll try different models
    # Start with Linear Regression (most FHE-friendly)
    print("Training Linear Regression model...")
    fhe_linear = LinearRegression()
    fhe_linear.fit(X_train, y_train)

    # Evaluate linear model
    y_pred_linear = fhe_linear.predict(X_test)
    linear_mae = mean_absolute_error(y_test, y_pred_linear)
    linear_r2 = r2_score(y_test, y_pred_linear)
    print(f"Linear Regression - MAE: {linear_mae:.3f}, R²: {linear_r2:.3f}")

    # Try Decision Tree (pruned for FHE compatibility)
    print("Training Decision Tree model...")
    fhe_dt = DecisionTreeRegressor(max_depth=4)  # Limit depth for FHE
    fhe_dt.fit(X_train, y_train)

    # Evaluate decision tree model
    y_pred_dt = fhe_dt.predict(X_test)
    dt_mae = mean_absolute_error(y_test, y_pred_dt)
    dt_r2 = r2_score(y_test, y_pred_dt)
    print(f"Decision Tree - MAE: {dt_mae:.3f}, R²: {dt_r2:.3f}")

    # Select best model based on MAE
    models = {
        'Linear': (fhe_linear, linear_mae, linear_r2, y_pred_linear),
        'Decision Tree': (fhe_dt, dt_mae, dt_r2, y_pred_dt)
    }

    best_model_name = min(models, key=lambda k: models[k][1])
    best_model, best_mae, best_r2, y_pred = models[best_model_name]

    print(f"\nBest FHE-compatible model for Aging Pace: {best_model_name}")
    print(f"MAE: {best_mae:.3f}, R²: {best_r2:.3f}")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Aging Pace')
    plt.ylabel('Predicted Aging Pace')
    plt.title(f'FHE Model: Actual vs Predicted Aging Pace - {best_model_name}')
    plt.tight_layout()
    plt.savefig(f"./plots/fhe_aging_pace_predictions_{best_model_name.lower().replace(' ', '_')}.png")
    plt.close()

    # Compile the model for FHE
    print("\nCompiling Aging Pace model for FHE...")

    # Use a small subset for compilation to save time
    X_sample = X_train[:100]

    # Configure quantization
    best_model.compile(X_sample)

    # Generate keys (dev only - in production this would be different)
    fhe_circuit = best_model.fhe_circuit

    print("Aging Pace model compiled for FHE!")

    # Save the FHE model and related components
    # joblib.dump(best_model, "fhe_aging_pace_model.joblib")
    # best_model.save("fhe_aging_pace_model")
    path_to_model = Path("./dev").resolve()
    if path_to_model.exists():
        shutil.rmtree(path_to_model)
    fhe_model_dev = FHEModelDev(path_to_model,best_model)
    fhe_model_dev.save(via_mlir=True)
    joblib.dump(data['aging_pace']['scaler'], "./plots/aging_pace_scaler.joblib")

    return best_model, best_model_name

def load_concrete_model(model_dir):
    for model_cls in [LinearRegression, DecisionTreeRegressor, RandomForestRegressor]:
        try:
            return model_cls.load(model_dir)
        except Exception:
            continue
    raise ValueError("Model type not recognized or supported.")

def compare_original_vs_fhe(data, fhe_model, model_type='bio_age'):
    """Compare the original sklearn model with the FHE model."""
    print(f"\n--- Comparing Original vs FHE Model for {model_type} ---")

    # Load original model
    if model_type == 'bio_age':
        # orig_model = joblib.load("biological_age_model.joblib")
        orig_model = load_concrete_model("fhe_bio_age_model")
        X_test = data['bio_age']['X_test']
        y_test = data['bio_age']['y_test']
        X_test_scaled = data['bio_age']['X_test_scaled']
    else:
        # orig_model = joblib.load("aging_pace_model.joblib")
        orig_model = load_concrete_model("aging_pace_model")
        X_test = data['aging_pace']['X_test']
        y_test = data['aging_pace']['y_test']
        X_test_scaled = data['aging_pace']['X_test_scaled']

    # Get predictions from original model
    y_pred_orig = orig_model.predict(X_test)
    orig_mae = mean_absolute_error(y_test, y_pred_orig)
    orig_r2 = r2_score(y_test, y_pred_orig)

    # Get predictions from FHE model
    y_pred_fhe = fhe_model.predict(X_test_scaled)
    fhe_mae = mean_absolute_error(y_test, y_pred_fhe)
    fhe_r2 = r2_score(y_test, y_pred_fhe)

    print(f"Original Model - MAE: {orig_mae:.3f}, R²: {orig_r2:.3f}")
    print(f"FHE Model - MAE: {fhe_mae:.3f}, R²: {fhe_r2:.3f}")

    # Calculate performance difference
    mae_diff = ((fhe_mae - orig_mae) / orig_mae) * 100
    r2_diff = ((orig_r2 - fhe_r2) / orig_r2) * 100 if orig_r2 > 0 else 0

    print(f"Performance difference:")
    print(f"MAE increase: {mae_diff:.2f}%")
    print(f"R² reduction: {r2_diff:.2f}%")

    # Plot comparison
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_orig, alpha=0.5, label='Original')
    plt.scatter(y_test, y_pred_fhe, alpha=0.5, label='FHE')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Original vs FHE Predictions')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_pred_orig, y_pred_fhe, alpha=0.5)
    min_val = min(y_pred_orig.min(), y_pred_fhe.min())
    max_val = max(y_pred_orig.max(), y_pred_fhe.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Original Model Predictions')
    plt.ylabel('FHE Model Predictions')
    plt.title('Prediction Correlation')

    plt.tight_layout()
    plt.savefig(f"model_comparison_{model_type}.png")
    plt.close()

def simulate_fhe_inference(fhe_model, model_type='bio_age'):
    """
    Simulate FHE inference with encrypted data.
    This demonstrates how the client-server interaction would work.
    """
    print(f"\n--- Simulating FHE Inference for {model_type} ---")

    # In a real-world scenario:
    # 1. Client has private data and public key
    # 2. Server has the model and evaluation key
    # 3. Client encrypts data and sends to server
    # 4. Server runs inference on encrypted data
    # 5. Server returns encrypted result
    # 6. Client decrypts result with private key

    # Generate sample input data
    if model_type == 'bio_age':
        scaler = joblib.load("./plots/bio_age_scaler.joblib")
    else:
        scaler = joblib.load("./plots/aging_pace_scaler.joblib")

    # Create a sample user (45-year-old with moderate health markers)
    sample_user = {
        'glucose_mg_dl': 95,
        'crp_mg_l': 1.8,
        'hdl_mg_dl': 52,
        'ldl_mg_dl': 125,
        'systolic_bp_mmhg': 125,
        'diastolic_bp_mmhg': 82,
        'bmi': 26.5,
        'hba1c_percent': 5.7,
        'albumin_g_dl': 4.1,
        'creatinine_mg_dl': 0.95,
        'alt_u_l': 25,
        'wbc_k_ul': 6.8,
        'sex': 1,  # Male
        'smoking_status': 0  # Never smoked
    }

    # Convert to DataFrame
    sample_df = pd.DataFrame([sample_user])

    # Scale the data
    sample_scaled = scaler.transform(sample_df)

    print("Sample user data:")
    for key, value in sample_user.items():
        print(f"  {key}: {value}")

    # Simulate encryption (in reality, this would be encrypted with FHE)
    print("\nSimulating encryption of user data...")

    # Simulate model evaluation on encrypted data
    print("Simulating server processing encrypted data...")

    # Get prediction (in cleartext for simulation)
    prediction = fhe_model.predict(sample_scaled)[0]

    print("Simulating decryption of result...")

    # Display result
    if model_type == 'bio_age':
        print(f"\nBiological Age Estimation: {prediction:.1f} years")
        chronological_age = 45
        difference = prediction - chronological_age
        status = "younger" if difference < 0 else "older"
        print(f"Chronological Age: {chronological_age} years")
        print(f"Biological vs Chronological difference: {abs(difference):.1f} years {status}")
    else:
        print(f"\nAging Pace Estimation: {prediction:.2f}")
        if prediction < 0.9:
            pace_desc = "slower than average"
        elif prediction > 1.1:
            pace_desc = "faster than average"
        else:
            pace_desc = "average"
        print(f"This indicates an aging pace that is {pace_desc}.")

    print("\nIn a real FHE deployment:")
    print("1. The user's biomarker data would be encrypted on the client-side")
    print("2. The server would process this encrypted data using FHE")
    print("3. The result would remain encrypted during the entire calculation")
    print("4. Only the user with the decryption key would see the final result")
    print("5. The service provider would gain no information about the user's data")

if __name__ == "__main__":
    # Load data
    data = load_data()

    # Train and convert biological age model
    bio_age_model, bio_model_name = train_and_convert_bio_age_model(data)

    # Train and convert aging pace model
    aging_pace_model, pace_model_name = train_and_convert_aging_pace_model(data)

    # Compare original vs FHE models
    # compare_original_vs_fhe(data, bio_age_model, 'bio_age')
    # compare_original_vs_fhe(data, aging_pace_model, 'aging_pace')

    # Simulate FHE inference
    simulate_fhe_inference(bio_age_model, 'bio_age')
    simulate_fhe_inference(aging_pace_model, 'aging_pace')

    print("\nFHE models created, evaluated, and simulated successfully!")