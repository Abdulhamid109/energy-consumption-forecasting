import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
df = pd.read_csv('powerDataset.csv')
df['Average_PowerConsumption'] = df[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']].mean(axis=1)

# Split the dataset into features (X) and target variable (y)
X = df[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 'Month', 'Day','Hour','Minute']]
y = df['Average_PowerConsumption']

# Reshape the target variable y
y = y.values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Choose Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=7, random_state=42)

# Train the model
y_train = np.ravel(y_train)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", score)
joblib.dump(rf_model, 'rf_model.pkl')

joblib.dump(scaler, 'scaler.pkl')