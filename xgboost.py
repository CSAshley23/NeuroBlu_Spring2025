from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import neuroblu as nb
scaler = MinMaxScaler()

def inverse_transform(pred_scaled):
    pred_log = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    return np.expm1(pred_log)

if __name__ == "__main__":
    df = nb.get_df("df_comorb_flags")

    # Columns to use
    categorical_columns = ['drug_concept_id', 'dose_unit_source_value', 'route_concept_id']
    numerical_columns = [
        'refills', 'quantity',
        'has_cvd', 'has_diabetes', 'has_sleep_apnea', 'has_dementia',
        'has_parkinsons', 'has_obesity', 'has_hyperlipidemia',
        'has_arrhythmia', 'has_epilepsy', 'has_autoimmune'
    ]

    #Encode categorical as label encoded (XGBoost-friendly)
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df[categorical_columns + numerical_columns].fillna(0)

    #Use log1p + scale for target 
    y_raw = df['days_supply'].values
    #compress large values and reduce outlier
    #helpful to learn multiplicative patterns linearly
    y = np.log1p(y_raw)
    y = scaler.fit_transform(y.reshape(-1,1)).flatten()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    #Train XGBoost 
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=0)
    model.fit(x_train.values, y_train)

    #Predict and evaluate 
    preds_scaled = model.predict(x_test.values)
    mse = mean_squared_error(y_test, preds_scaled)
    r2 = r2_score(y_test, preds_scaled)

    #show predictions
    preds_days_supply = inverse_transform(preds_scaled)
    true_days_supply = inverse_transform(y_test)
    

    print(f"XGBoost - Mean Squared Error: {mse:.2f}")
    print(f"XGBoost - R² Score: {r2:.4f}")

    # Print first 10 predictions
    for i in range(10):
        print(f"Predicted: {preds_days_supply[i]:.1f} days, Actual: {true_days_supply[i]:.1f} days")
