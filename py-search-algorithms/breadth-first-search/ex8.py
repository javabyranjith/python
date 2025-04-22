# EXPERIMENT - 8

# Implementation Of ensemble techniques
#pip install xgboost


import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("C:/Users/java2/github/python/py-search-algorithms/breadth-first-search/train_data.csv")
target = df["target"]
train = df.drop("target", axis=1)

# Split data
train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=1 - train_ratio)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

# Initialize models
model_1 = LinearRegression()
model_2 = xgb.XGBRegressor()
model_3 = RandomForestRegressor()

# Train and predict using Linear Regression
model_1.fit(x_train, y_train)
val_pred_1 = model_1.predict(x_val)
test_pred_1 = model_1.predict(x_test)

# Train and predict using XGBoost
model_2.fit(x_train, y_train)
val_pred_2 = model_2.predict(x_val)
test_pred_2 = model_2.predict(x_test)

# Train and predict using RandomForest
model_3.fit(x_train, y_train)
val_pred_3 = model_3.predict(x_val)
test_pred_3 = model_3.predict(x_test)

# Convert predictions to DataFrame for better visualization
val_pred_1 = pd.DataFrame(val_pred_1, columns=['Model_1_Pred'])
test_pred_1 = pd.DataFrame(test_pred_1, columns=['Model_1_Pred'])

val_pred_2 = pd.DataFrame(val_pred_2, columns=['Model_2_Pred'])
test_pred_2 = pd.DataFrame(test_pred_2, columns=['Model_2_Pred'])

val_pred_3 = pd.DataFrame(val_pred_3, columns=['Model_3_Pred'])
test_pred_3 = pd.DataFrame(test_pred_3, columns=['Model_3_Pred'])

# Combine validation and test predictions into a single DataFrame
df_val = pd.concat([x_val, val_pred_1, val_pred_2, val_pred_3], axis=1)
df_test = pd.concat([x_test, test_pred_1, test_pred_2, test_pred_3], axis=1)

# Train the final model on validation predictions
final_model = LinearRegression()
final_model.fit(df_val, y_val)

# Make final predictions on the test set
final_pred = final_model.predict(df_test)

# Calculate and print the mean squared error of the final model's predictions
mse = mean_squared_error(y_test, final_pred)
print(f'Mean Squared Error of final model: {mse}')


