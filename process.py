import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Read the CSV file into a pandas DataFrame
df = pd.read_csv('yield_with_sine_cosine.csv')

# Convert the 'Date' column to datetime format (if it's still in the DataFrame)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Sort by date to ensure proper lag creation
df = df.sort_values(by='Date')

# Create lagged features
num_lags = 3  # Number of lagged features to create
for lag in range(1, num_lags + 1):
    df[f'Weight_lag_{lag}'] = df['Weight'].shift(lag)

# Drop rows with NaN values (which will be created due to shifting)
df = df.dropna()

# Define predictors (sine and cosine columns, and lagged features) and target (Weight)
predictors = [col for col in df.columns if '_sine' in col or '_cosine' in col] + [f'Weight_lag_{lag}' for lag in range(1, num_lags + 1)]
X = df[predictors]
y = df['Weight']

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a machine learning model (Random Forest Regressor in this case)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 5: Plot prediction vs. true values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Diagonal line
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Prediction vs. True Values (Random Forest with Lagged Features)')
plt.grid(True)
plt.show()
