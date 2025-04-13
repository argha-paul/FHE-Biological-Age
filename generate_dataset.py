import numpy as np
import pandas as pd
# from sklearn.datasets import make_regression

def generate_biomarker_dataset(n_samples=1000, random_state=42):
    """
    Generate a synthetic dataset of biomarkers related to biological aging.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing biomarkers and target variables
    """
    np.random.seed(random_state)

    # Define biomarker ranges and correlations with biological age
    biomarkers = {
        'glucose_mg_dl': {'mean': 95, 'std': 15, 'age_corr': 0.4},
        'crp_mg_l': {'mean': 1.5, 'std': 2.0, 'age_corr': 0.5},
        'hdl_mg_dl': {'mean': 55, 'std': 12, 'age_corr': -0.3},
        'ldl_mg_dl': {'mean': 120, 'std': 30, 'age_corr': 0.35},
        'systolic_bp_mmhg': {'mean': 120, 'std': 15, 'age_corr': 0.45},
        'diastolic_bp_mmhg': {'mean': 80, 'std': 10, 'age_corr': 0.35},
        'bmi': {'mean': 25, 'std': 4, 'age_corr': 0.2},
        'hba1c_percent': {'mean': 5.5, 'std': 0.7, 'age_corr': 0.4},
        'albumin_g_dl': {'mean': 4.3, 'std': 0.4, 'age_corr': -0.25},
        'creatinine_mg_dl': {'mean': 0.9, 'std': 0.2, 'age_corr': 0.3},
        'alt_u_l': {'mean': 22, 'std': 10, 'age_corr': 0.15},
        'wbc_k_ul': {'mean': 6.5, 'std': 1.8, 'age_corr': 0.2}
    }

    # Create chronological age (25-85)
    chron_age = np.random.uniform(25, 85, n_samples)

    # Generate biological age with random deviation from chronological age
    # People who are healthier can be up to 20 years younger biologically
    # People who are less healthy can be up to 20 years older biologically
    bio_age_deviation = np.random.normal(0, 7, n_samples)
    bio_age = np.clip(chron_age + bio_age_deviation, 18, 105)

    # Generate aging pace: how many years of biological aging per chronological year
    # 1.0 means normal pace, <1.0 means slower, >1.0 means faster
    aging_pace = np.clip(1.0 + bio_age_deviation/35, 0.5, 1.5)

    # Create DataFrame with chronological and biological ages
    df = pd.DataFrame({
        'chronological_age': chron_age,
        'biological_age': bio_age,
        'aging_pace': aging_pace,
    })

    # Create biomarkers based on biological age with noise
    for biomarker, params in biomarkers.items():
        # Base biomarker value
        base_values = np.random.normal(params['mean'], params['std'], n_samples)

        # Add age-related trend
        age_effect = params['age_corr'] * (bio_age - np.mean(bio_age)) / 10

        # Final biomarker values with constraints to realistic ranges
        df[biomarker] = base_values + age_effect * params['std']

        # Add some non-linearity
        if params['age_corr'] > 0:
            df[biomarker] += 0.1 * params['std'] * (bio_age / 50) ** 2

    # Ensure realistic ranges
    df['glucose_mg_dl'] = np.clip(df['glucose_mg_dl'], 60, 180)
    df['crp_mg_l'] = np.clip(df['crp_mg_l'], 0.1, 15)
    df['hdl_mg_dl'] = np.clip(df['hdl_mg_dl'], 20, 100)
    df['ldl_mg_dl'] = np.clip(df['ldl_mg_dl'], 40, 220)
    df['systolic_bp_mmhg'] = np.clip(df['systolic_bp_mmhg'], 90, 180)
    df['diastolic_bp_mmhg'] = np.clip(df['diastolic_bp_mmhg'], 50, 110)
    df['bmi'] = np.clip(df['bmi'], 16, 45)
    df['hba1c_percent'] = np.clip(df['hba1c_percent'], 4.0, 9.0)
    df['albumin_g_dl'] = np.clip(df['albumin_g_dl'], 2.5, 5.5)
    df['creatinine_mg_dl'] = np.clip(df['creatinine_mg_dl'], 0.4, 2.5)
    df['alt_u_l'] = np.clip(df['alt_u_l'], 5, 100)
    df['wbc_k_ul'] = np.clip(df['wbc_k_ul'], 2.5, 15)

    # Add categorical variable: sex (biological)
    df['sex'] = np.random.choice([0, 1], size=n_samples)  # 0=female, 1=male

    # Add smoking status
    df['smoking_status'] = np.random.choice([0, 1, 2], size=n_samples,
                                            p=[0.6, 0.25, 0.15])  # 0=never, 1=former, 2=current

    # Make smokers have slightly worse biomarkers
    smoking_effect = np.where(df['smoking_status'] == 0, 0,
                              np.where(df['smoking_status'] == 1, 0.5, 1.0))
    df['glucose_mg_dl'] += smoking_effect * 3
    df['crp_mg_l'] += smoking_effect * 0.8
    df['hdl_mg_dl'] -= smoking_effect * 3
    df['systolic_bp_mmhg'] += smoking_effect * 5

    return df

if __name__ == "__main__":
    # Generate dataset with 1500 samples
    biomarker_data = generate_biomarker_dataset(n_samples=1500)

    # Save to CSV
    biomarker_data.to_csv("biomarker_dataset.csv", index=False)

    print(f"Dataset generated with {len(biomarker_data)} samples and {biomarker_data.shape[1]} features.")
    print("\nFeature statistics:")
    print(biomarker_data.describe().round(2))