import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the CSV files
df_weight = pd.read_csv('yield.csv', parse_dates=['Date'])
df_oil = pd.read_csv('oil_price_brent.csv', parse_dates=['Date'])

# Clean the oil_price data by replacing invalid strings (e.g., '.') with NaN
df_oil['price'] = pd.to_numeric(df_oil['price'], errors='coerce')

# Remove rows where oil price is missing (i.e., NaN values after cleaning)
df_oil = df_oil.dropna(subset=['price'])

# Resample oil price data to monthly average
df_oil.set_index('Date', inplace=True)
df_oil_monthly = df_oil.resample('M').mean().reset_index()

# Merge the monthly oil price data with the weight data on the Date column
df_weight = df_weight.sort_values('Date')

# Align monthly oil prices with weekly weight data by forward filling
df_merged = pd.merge_asof(df_weight, df_oil_monthly, left_on='Date', right_on='Date', direction='backward')

# Forward fill missing values after merge to apply monthly oil price to each week
df_merged = df_merged.ffill()

# Ensure no NaN values remain after forward fill
df_merged = df_merged.dropna()

# Scale Weight and Monthly Average Oil Price to the range [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
df_merged[['Weight_scaled', 'Oil_Price_scaled']] = scaler.fit_transform(df_merged[['Weight', 'price']])

# Calculate the correlation between scaled Weight and scaled Oil Price
correlation = df_merged['Weight_scaled'].corr(df_merged['Oil_Price_scaled'])
print(f"Correlation between Scaled Weight and Scaled Monthly Average Oil Price: {correlation:.2f}")

# Plot the scaled Weight and Monthly Average Oil Price aligned with weekly data
plt.figure(figsize=(12, 6))

# Plot scaled Weight
plt.plot(df_merged['Date'], df_merged['Weight_scaled'], 'b-o', label='Scaled Weight (Weekly)')

# Plot scaled Monthly Average Oil Price
plt.plot(df_merged['Date'], df_merged['Oil_Price_scaled'], 'r--', label='Scaled Oil Price (Monthly Avg)')

# Add correlation text to the plot
plt.text(df_merged['Date'].iloc[0], df_merged['Weight_scaled'].max(), 
         f'Correlation: {correlation:.2f}', 
         color='black', fontsize=12, ha='left')

# Customize the plot
plt.title('Scaled Weight (Weekly) and Scaled Monthly Average Oil Price (Brent)')
plt.xlabel('Date')
plt.ylabel('Scaled Value')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
