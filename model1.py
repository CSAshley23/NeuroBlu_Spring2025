from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import numpy as np
import neuroblu as nb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def generate_onehot_encoded_dataset(data, categorical_columns, numerical_columns):
    #First convert the categorical columns 
    #Sparse output set to False for a smaller dataset
    #Drop the first column to reduce redudant data
    #Handle unknown will encode unknown categories as all 0's
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

    #Transform all categorical coumns
    encoded_cats = encoder.fit_transform(data[categorical_columns])
    #Create new categories for the new categories generated from above
    #encoded_cat_names = encoder.get_feature_names_out(categorical_columns)

    scalar = preprocessing.MinMaxScaler()
    #Convert the numerical features to a range 0 to 1
    numerical_features = scalar.fit_transform(data[numerical_columns].fillna(0).values)

    #Combine both transformed features into a features variable
    #Below is a sample of what the data combined looks like
    #Each of these represent one row of data
    # {
    # "age_scaled": 0.3,
    # "quantity_scaled": 0.6,
    # "gender_8507": 1,
    # "gender_8532": 0
    # }
    features = np.hstack((numerical_features, encoded_cats))

    # all_feature_names = numerical_columns + list(encoded_cat_names)
    # actual_feature_names=[]
    # Handle target variable ('truth') with clipping and transformation
    truth = data['days_supply'].values
    print("Min:", truth.min())

    #Maybe add this back in when the data is better
    truth = np.log1p(truth)  # Apply log transformation
    truth = preprocessing.MinMaxScaler().fit_transform(truth.reshape(-1, 1)).flatten()  # Rescale to 0-1

    return features, truth


if(__name__ =="__main__"):
    df = nb.get_df('df_comorb_flags')
    categorical_columns = ['drug_concept_id','dose_unit_source_value','route_concept_id']
    numerical_columns = [
    'refills', 'quantity',
    'has_cvd', 'has_diabetes', 'has_sleep_apnea', 'has_dementia',
    'has_parkinsons', 'has_obesity', 'has_hyperlipidemia',
    'has_arrhythmia', 'has_epilepsy', 'has_autoimmune'
    ]

    features, truth = generate_onehot_encoded_dataset(df, categorical_columns, numerical_columns)
    # print("This is features", features)
    # print("This is the truth values", truth)

    x_train, x_test, y_train, y_test = train_test_split(features, truth, test_size=0.15, random_state=0)

    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(x_train, y_train)
    predictions_ridge = model_ridge.predict(x_test)
    
    mse_ridge = mean_squared_error(y_test, predictions_ridge)
    r2_ridge = r2_score(y_test, predictions_ridge)
    print(f"Ridge Regression - Mean Squared Error: {mse_ridge:.2f}")
    print(f"Ridge Regression - R² Score: {r2_ridge:.2f}")
    
    




    
    

        
    
