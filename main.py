import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

#1. Load the dataset
df_housing= pd.read_csv('housing.csv')

#2. Create a stratified test set
df_housing['income_cat'] = pd.cut(df_housing['median_income'],
                                 bins = [0,1.5,3.0,4.5,6.0,np.inf],
                                 labels = [1,2,3,4,5])


split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)                                
for train_index, test_index in split.split(df_housing,df_housing['income_cat']):
    strat_train_set = df_housing.loc[train_index].drop("income_cat",axis =1)
    strat_test_set = df_housing.loc[test_index].drop("income_cat",axis =1)

# We'll work on the copy of training dataset
df1 = strat_train_set.copy()

#3. Separate features and labels
housing_labels = df1['median_house_value'].copy()
housing_features = df1.drop("median_house_value",axis= 1)

#4. List the Numerical and Categorical cols
numerical_attributes= housing_features.drop("ocean_proximity",axis = 1).columns.tolist()
categorical_attribute = ['ocean_proximity']

#5. Pipeline
# a. For Numerical cols
num_pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])
# b. For Categorical cols
cat_pipeline = Pipeline([
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])
# c. Construct the full-pipeline
full_pipeline = ColumnTransformer([
    ("numerical",num_pipeline, numerical_attributes),
    ("categorical",cat_pipeline,categorical_attribute)
])

#6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing_features)
# print(housing_prepared)
# print(housing_prepared.shape)

#7. Train the Model
# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_pred= lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels,lin_pred)
# print(f"The root mean squared error for Linear Regression is {lin_rmse}")
lin_rmses = - cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv= 10)
print(pd.Series(lin_rmses).describe())

# Decision Tree Model
DecT_reg = DecisionTreeRegressor()
DecT_reg.fit(housing_prepared, housing_labels)
DecT_pred= DecT_reg.predict(housing_prepared)
DecT_rmse = root_mean_squared_error(housing_labels,DecT_pred)
# print(f"The root mean squared error for Decision Tree Regressor is {DecT_rmse}")
DecT_rmses = - cross_val_score(DecT_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv= 10)
print(pd.Series(DecT_rmses).describe())
print(np.mean(DecT_rmses))
print(DecT_rmses)

# # Random Forest Model
# RandomF_reg = RandomForestRegressor()
# RandomF_reg.fit(housing_prepared, housing_labels)
# RandomF_pred= RandomF_reg.predict(housing_prepared)
# RandomF_rmse = root_mean_squared_error(housing_labels,RandomF_pred)
# # print(f"The root mean squared error for Random Forest Regressor is {RandomF_rmse}")
# RandomF_rmses = - cross_val_score(RandomF_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv= 10)
# print(pd.Series(RandomF_rmses).describe())