# Home Value Prediction
# Predicts house prices based on factors like average income and crime rate using linear regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
np.random.seed(42)
data = {
    'avg_income': np.random.normal(50000, 10000, 1000),
    'crime_rate': np.random.uniform(0, 10, 1000),
    'house_price': np.zeros(1000)
}
for i in range(1000):
    data['house_price'][i] = (data['avg_income'][i] * 0.005 - data['crime_rate'][i] * 1000 + 
                              np.random.normal(0, 5000))

df = pd.DataFrame(data)

# Prepare features and target
X = df[['avg_income', 'crime_rate']]
y = df['house_price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example prediction
sample = np.array([[60000, 2.5]])
pred_price = model.predict(sample)
print(f"Predicted house price: ${pred_price[0]:.2f}")