# model_training_house_prices.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the Ames Housing dataset
data = pd.read_csv('AmesHousing.csv')

# Select relevant features and target variable
selected_features = [
    'Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built',
    'Gr Liv Area', 'Total Bsmt SF', 'Garage Area', 'Full Bath',
    'Half Bath', 'Bedroom AbvGr', 'TotRms AbvGrd'
]
target = 'SalePrice'

# Prepare the dataset
X = data[selected_features]
y = data[target]

# Handle missing values by filling them with the mean
X = X.fillna(X.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f"Root Mean Squared Error: {rmse}")

# Save the trained model
joblib.dump(model, 'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl'.")
