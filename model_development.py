import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)
import os

# Ensure the directory exists
os.makedirs("./plots", exist_ok=True)

def load_and_prepare_data(file_path="biomarker_dataset.csv"):
    """
    Load and prepare the biomarker dataset for modeling.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing biomarker data

    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy arrays
        Training and testing data splits
    feature_names : list
        List of feature names
    """
    # Load data
    df = pd.read_csv(file_path)

    # Display information about the data
    print(f"Dataset loaded with {df.shape[0]} samples and {df.shape[1]} columns")

    # Separate features and targets
    X = df.drop(['chronological_age', 'biological_age', 'aging_pace'], axis=1)
    y_bio_age = df['biological_age']
    y_aging_pace = df['aging_pace']

    feature_names = X.columns.tolist()

    # Train-test split
    X_train_bio, X_test_bio, y_train_bio, y_test_bio = train_test_split(
        X, y_bio_age, test_size=0.2, random_state=42
    )

    X_train_pace, X_test_pace, y_train_pace, y_test_pace = train_test_split(
        X, y_aging_pace, test_size=0.2, random_state=42
    )

    return {
        'bio_age': (X_train_bio, X_test_bio, y_train_bio, y_test_bio),
        'aging_pace': (X_train_pace, X_test_pace, y_train_pace, y_test_pace),
        'feature_names': feature_names,
        'full_data': df
    }

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data.

    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : numpy array
        Test features
    y_test : numpy array
        Test targets

    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    y_pred = model.predict(X_test)

    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    }

    return metrics, y_pred

def plot_feature_importance(model, feature_names, title):
    """
    Plot feature importance from a model.

    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    title : str
        Plot title
    """
    if hasattr(model, 'feature_importances_'):
        # For models that have feature_importances_ attribute
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models with coefficients
        importances = np.abs(model.coef_)
    else:
        print("Model doesn't have feature importances or coefficients")
        return

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance - {title}")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f"./plots/feature_importance_{title.lower().replace(' ', '_')}.png")
    plt.close()

def train_biological_age_model(data_dict):
    """
    Train a biological age prediction model.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data splits

    Returns:
    --------
    best_model : sklearn model
        Best performing trained model
    """
    X_train, X_test, y_train, y_test = data_dict['bio_age']
    feature_names = data_dict['feature_names']

    # Define models to evaluate
    models = {
        'Random Forest': RandomForestRegressor(max_depth=5),
        'LinearRegression':LinearRegression(),
        'RandomForestRegression':RandomForestRegressor(n_estimators=5, max_depth=4)
    }

    best_score = float('inf')
    best_model_name = None
    best_model = None
    results = {}

    # Evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -np.mean(cv_scores)

        # Fit on full training data
        pipeline.fit(X_train, y_train)

        # Evaluate on test data
        metrics, y_pred = evaluate_model(pipeline, X_test, y_test)

        results[name] = {
            'cv_mae': cv_mae,
            'test_metrics': metrics,
            'model': pipeline
        }

        print(f"{name} - CV MAE: {cv_mae:.3f}, Test MAE: {metrics['MAE']:.3f}, "
              f"Test RMSE: {metrics['RMSE']:.3f}, Test R²: {metrics['R²']:.3f}")

        # Update best model if this one is better
        if metrics['MAE'] < best_score:
            best_score = metrics['MAE']
            best_model_name = name
            best_model = pipeline

    print(f"\nBest model: {best_model_name} with MAE: {best_score:.3f}")

    # Plot actual vs predicted for best model
    y_pred = best_model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Biological Age')
    plt.ylabel('Predicted Biological Age')
    plt.title(f'Actual vs Predicted - {best_model_name}')
    plt.tight_layout()
    plt.savefig("./plots/biological_age_predictions.png")
    plt.close()

    # Plot feature importance for best model
    try:
        plot_feature_importance(best_model.named_steps['model'],
                                feature_names,
                                "Biological Age Model")
    except:
        print("Could not plot feature importance")

    # Save the best model
    joblib.dump(best_model, "./plots/biological_age_model.joblib")

    return best_model

def train_aging_pace_model(data_dict):
    """
    Train an aging pace prediction model.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data splits

    Returns:
    --------
    best_model : sklearn model
        Best performing trained model
    """
    X_train, X_test, y_train, y_test = data_dict['aging_pace']
    feature_names = data_dict['feature_names']

    # Define models to evaluate
    models = {
        'Ridge': Ridge(alpha=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    }

    best_score = float('inf')
    best_model_name = None
    best_model = None
    results = {}

    # Evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name} for Aging Pace...")

        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -np.mean(cv_scores)

        # Fit on full training data
        pipeline.fit(X_train, y_train)

        # Evaluate on test data
        metrics, y_pred = evaluate_model(pipeline, X_test, y_test)

        results[name] = {
            'cv_mae': cv_mae,
            'test_metrics': metrics,
            'model': pipeline
        }

        print(f"{name} - CV MAE: {cv_mae:.3f}, Test MAE: {metrics['MAE']:.3f}, "
              f"Test RMSE: {metrics['RMSE']:.3f}, Test R²: {metrics['R²']:.3f}")

        # Update best model if this one is better
        if metrics['MAE'] < best_score:
            best_score = metrics['MAE']
            best_model_name = name
            best_model = pipeline

    print(f"\nBest model for Aging Pace: {best_model_name} with MAE: {best_score:.3f}")

    # Plot actual vs predicted for best model
    y_pred = best_model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Aging Pace')
    plt.ylabel('Predicted Aging Pace')
    plt.title(f'Actual vs Predicted Aging Pace - {best_model_name}')
    plt.tight_layout()
    plt.savefig("./plots/aging_pace_predictions.png")
    plt.close()

    # Plot feature importance for best model
    try:
        plot_feature_importance(best_model.named_steps['model'],
                                feature_names,
                                "Aging Pace Model")
    except:
        print("Could not plot feature importance")

    # Save the best model
    joblib.dump(best_model, "./plots/aging_pace_model.joblib")

    return best_model

def analyze_data(data_dict):
    """
    Perform exploratory data analysis on the dataset.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing the full dataset
    """
    df = data_dict['full_data']

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm',
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig("./plots/correlation_heatmap.png")
    plt.close()

    # Biological Age vs Chronological Age
    plt.figure(figsize=(10, 6))
    plt.scatter(df['chronological_age'], df['biological_age'], alpha=0.5)
    plt.plot([20, 90], [20, 90], 'r--')  # Identity line
    plt.xlabel('Chronological Age')
    plt.ylabel('Biological Age')
    plt.title('Biological Age vs Chronological Age')
    plt.tight_layout()
    plt.savefig("./plots/bio_vs_chrono_age.png")
    plt.close()

    # Distribution of aging pace
    plt.figure(figsize=(10, 6))
    sns.histplot(df['aging_pace'], kde=True)
    plt.axvline(1.0, color='red', linestyle='--')
    plt.xlabel('Aging Pace')
    plt.ylabel('Count')
    plt.title('Distribution of Aging Pace')
    plt.tight_layout()
    plt.savefig("./plots/aging_pace_distribution.png")
    plt.close()

    # Top correlations with biological age
    bio_age_corr = df.corr()['biological_age'].sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=bio_age_corr.values[1:11], y=bio_age_corr.index[1:11])
    plt.xlabel('Correlation')
    plt.ylabel('Features')
    plt.title('Top Correlations with Biological Age')
    plt.tight_layout()
    plt.savefig("./plots/bio_age_correlations.png")
    plt.close()

if __name__ == "__main__":
    # Load and prepare data
    data = load_and_prepare_data()

    # Analyze dataset
    analyze_data(data)

    # Train biological age model
    bio_age_model = train_biological_age_model(data)

    # Train aging pace model
    aging_pace_model = train_aging_pace_model(data)

    print("\nModels trained and saved successfully!")