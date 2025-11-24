import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def Build_Pipeline(numerical_attributes, categorical_attribute):
    #5. Pipeline
# a. For Numerical cols
    num_pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
    ])
# b. For Categorical cols
    cat_pipeline = Pipeline([
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
]   )
# c. Construct the full-pipeline
    full_pipeline = ColumnTransformer([
    ("numerical",num_pipeline, numerical_attributes),
    ("categorical",cat_pipeline,categorical_attribute)
]   )
    return full_pipeline
    

if not os.path.exists(MODEL_FILE):
    # Train the Model.
    df_housing= pd.read_csv('housing.csv')
    df_housing['income_cat'] = pd.cut(df_housing['median_income'],
                                    bins = [0,1.5,3.0,4.5,6.0,np.inf],
                                    labels = [1,2,3,4,5])


    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)                                
    for train_index, test_index in split.split(df_housing,df_housing['income_cat']):
        df_housing.loc[test_index].drop("income_cat",axis=1).to_csv("input.csv",index = False)
        df_housing = df_housing.loc[train_index].drop("income_cat",axis =1)
        
        
    housing_labels = df_housing['median_house_value'].copy()
    housing_features = df_housing.drop("median_house_value",axis= 1)

    numerical_attributes= housing_features.drop("ocean_proximity",axis = 1).columns.tolist()
    categorical_attribute = ['ocean_proximity']
    
    pipeline = Build_Pipeline(numerical_attributes, categorical_attribute)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model is trained. Congrats!")

else:
    # Do Inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)
    Predictions = model.predict(transformed_input) 
    input_data['median_house_value'] = Predictions
    input_data.to_csv("output.csv", index = False)
    print("Inference has been Completed, results saved to output.csv Enjoy :)")
