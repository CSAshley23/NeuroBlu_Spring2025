import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import neuroblu as nb

# Temporary patches to avoid NumPy compatibility issues
np.bool = bool
np.int = int

# Initialize MinMaxScaler
scaler = MinMaxScaler()

def inverse_transform(pred_scaled):
    pred_log = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return np.expm1(pred_log)

if __name__ == "__main__":
    # Load data
    df = nb.get_df("df_comorb_flags")

    # Columns to use
    categorical_columns = ['drug_concept_id', 'dose_unit_source_value', 'route_concept_id']
    numerical_columns = [
        'refills', 'quantity',
        'has_cvd', 'has_diabetes', 'has_sleep_apnea', 'has_dementia',
        'has_parkinsons', 'has_obesity', 'has_hyperlipidemia',
        'has_arrhythmia', 'has_epilepsy', 'has_autoimmune'
    ]

    # Encode categorical features
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Combine categorical and numerical columns
    X = df[categorical_columns + numerical_columns].fillna(0)

    # Ensure all columns are numeric and clean data
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

    # Target variable
    y_raw = df['days_supply'].values
    y = np.log1p(y_raw)
    y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    # Convert to NumPy arrays (avoiding Pandas-specific issues)
    x_train_numpy = x_train.values.astype(np.float64)
    x_test_numpy = x_test.values.astype(np.float64)

    # Train XGBoost model
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=0)
    model.fit(x_train_numpy, y_train)

    # Predict and evaluate
    preds_scaled = model.predict(x_test_numpy)
    mse = mean_squared_error(y_test, preds_scaled)
    r2 = r2_score(y_test, preds_scaled)

    # Inverse transform predictions for reporting
    preds_days_supply = inverse_transform(preds_scaled)
    true_days_supply = inverse_transform(y_test)

    print(f"XGBoost - Mean Squared Error: {mse:.2f}")
    print(f"XGBoost - R² Score: {r2:.4f}")


    # Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(x_test_numpy)

    # Debugging output to ensure SHAP values are valid
    print(f"SHAP values shape: {np.shape(shap_values)}")
    
    # Calculate mean absolute SHAP values for tornado plot
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Feature names
    feature_names = x_test.columns if isinstance(x_test, pd.DataFrame) else [f"Feature {i}" for i in range(len(mean_shap_values))]

    # Create a DataFrame for better visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean SHAP Value': mean_shap_values
    })

    # Sort features by importance
    feature_importance = feature_importance.sort_values(by='Mean SHAP Value', ascending=False)

    # Plot the tornado plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Mean SHAP Value'], color='skyblue')
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Tornado Plot: Feature Importance', fontsize=14)
    plt.gca().invert_yaxis()  # Ensure the most important features appear at the top
    plt.tight_layout()
    plt.savefig("tornado_plot.png")
    print("Plot saved as tornado_plot.png")



   