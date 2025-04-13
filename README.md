# Biological Age Estimation using Fully Homomorphic Encryption (FHE)

This project implements secure machine learning models for biological age and aging pace estimation using Fully Homomorphic Encryption (FHE). By leveraging Zama's Concrete ML library, the system enables privacy-preserving health analytics where sensitive biomarker data remains encrypted throughout the prediction process.

## Table of Contents
- [Project Objectives](#project-objectives)
- [Background](#background)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Implementation](#technical-implementation)
    - [Data Processing](#data-processing)
    - [Model Development](#model-development)
    - [FHE Conversion](#fhe-conversion)
    - [Inference Simulation](#inference-simulation)
- [Model Performance](#model-performance)
- [Privacy Benefits](#privacy-benefits)
- [Limitations](#limitations)
- [Future Work](#future-work)

## Project Objectives

The primary objectives of this project are:

1. Develop machine learning models to predict biological age and aging pace from biomarker data
2. Convert these models to FHE-compatible versions using Concrete ML
3. Demonstrate privacy-preserving biological age estimation where user data remains encrypted
4. Compare the performance of traditional ML models with their FHE-compatible counterparts

## Background

Biological age estimation has significant clinical and research applications, potentially serving as a better indicator of health status than chronological age. However, the biomarker data required for these predictions is highly sensitive personal health information. Using Fully Homomorphic Encryption allows computations to be performed on encrypted data without ever decrypting it, preserving user privacy while still providing valuable health insights.

## Features

- Prediction of biological age from common biomarker data
- Estimation of aging pace (how quickly someone is aging relative to the average)
- Privacy-preserving inference using FHE
- Multiple model architectures (Linear Regression, Decision Tree, Random Forest)
- Visualization of model performance and feature importance
- Simulation of secure client-server interaction


## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/FHE-Biological-Age.git
   cd FHE-Biological-Age
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

The project expects a CSV file named `biomarker_dataset.csv` with the following columns:
- `chronological_age`: Actual age in years
- `biological_age`: Reference biological age (for training/evaluation)
- `aging_pace`: Reference aging pace (for training/evaluation)
- Biomarker columns: glucose_mg_dl, crp_mg_l, hdl_mg_dl, ldl_mg_dl, systolic_bp_mmhg, diastolic_bp_mmhg, bmi, hba1c_percent, albumin_g_dl, creatinine_mg_dl, alt_u_l, wbc_k_ul, sex, smoking_status

### Running the Pipeline

1. Train the traditional ML models:
   ```
   python model_development.py
   ```

2. Convert to FHE-compatible models and simulate inference:
   ```
   python fhe_conversion.py
   ```

## Technical Implementation

### Data Processing

The model development pipeline includes the following data processing steps:

1. Loading the biomarker dataset from CSV
2. Separating features (biomarkers) from targets (biological age, aging pace)
3. Train-test splitting (80% training, 20% testing)
4. Feature standardization using `StandardScaler`

### Model Development

The `model_development.py` script:

1. Performs exploratory data analysis (EDA) to understand relationships between biomarkers and biological age
2. Trains multiple regression models for biological age prediction:
    - Random Forest Regressor
    - Linear Regression
    - Other model types (Ridge, Lasso, etc.)
3. Evaluates models using cross-validation and test set metrics (MAE, RMSE, R²)
4. Selects the best-performing model based on MAE
5. Visualizes feature importance and actual vs. predicted values
6. Repeats the process for aging pace prediction

Key functions:
- `load_and_prepare_data()`: Data loading and preprocessing
- `evaluate_model()`: Model evaluation metrics
- `train_biological_age_model()`: Training and evaluation for biological age
- `train_aging_pace_model()`: Training and evaluation for aging pace
- `analyze_data()`: Exploratory data analysis and visualization
- `plot_feature_importance()`: Visualization of feature importance

### FHE Conversion

The `fhe_conversion.py` script:

1. Trains FHE-compatible versions of the models using Concrete ML:
    - Linear Regression (most FHE-friendly)
    - Decision Tree (with limited depth for FHE compatibility)
    - Random Forest (with fewer trees and limited depth)
2. Evaluates and compares FHE models with traditional ML models
3. Selects the best FHE-compatible model based on MAE
4. Compiles models for FHE execution
5. Simulates secure inference with encrypted data

Key functions:
- `load_data()`: Data loading with standardization
- `train_and_convert_bio_age_model()`: FHE model for biological age
- `train_and_convert_aging_pace_model()`: FHE model for aging pace
- `compare_original_vs_fhe()`: Comparison between traditional and FHE models
- `simulate_fhe_inference()`: FHE inference simulation

### Inference Simulation

The project demonstrates a simulated secure inference process:

1. Client prepares biomarker data
2. Data is scaled using the stored scaler
3. (In real-world FHE): Client encrypts data with their public key
4. Server receives encrypted data and runs inference without decryption
5. Server returns encrypted prediction
6. Client decrypts results using their private key

## Model Performance

The system evaluates model performance using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (R²)

FHE-compatible models typically show some performance degradation compared to traditional ML models due to:
- Limited model complexity (shallower trees, fewer estimators)
- Integer-based computation instead of floating point
- Quantization effects

## Privacy Benefits

Using FHE for biological age estimation provides several privacy advantages:

1. **Data Confidentiality**: Biomarker data remains encrypted throughout processing
2. **Service Provider Blindness**: The service provider never sees the user's health data
3. **Regulatory Compliance**: Helps meet HIPAA, GDPR, and other health data regulations
4. **Zero-Knowledge Inference**: User gets predictions without revealing sensitive information

## Limitations

Current limitations of the FHE approach include:

1. **Computational Overhead**: FHE operations are more computationally expensive
2. **Model Constraints**: Not all ML architectures are FHE-friendly (e.g., deep neural networks)
3. **Precision Loss**: Integer-based computation can reduce prediction accuracy
4. **Development Complexity**: FHE requires specialized knowledge and tools

## Future Work

Potential improvements and extensions:

1. Explore advanced FHE-friendly model architectures
2. Implement secure multi-party computation for federated learning
3. Develop a web service interface for encrypted predictions
4. Expand to additional health metrics beyond biological age
5. Optimize FHE parameter selection for better performance