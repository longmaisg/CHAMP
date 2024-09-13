import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the CSV files
df_weight = pd.read_csv('yield.csv', parse_dates=['Date'])

# Load PMI data from pmi_china.txt
df_pmi = pd.read_csv('pmi_china.txt', sep='\t', names=['Date', 'Value'], skiprows=1)

# Parse the dates in the PMI data
df_pmi['Date'] = pd.to_datetime(df_pmi['Date'], format='%B %d, %Y')

# Ensure both dataframes are sorted by Date
df_weight = df_weight.sort_values('Date')
df_pmi = df_pmi.sort_values('Date')

# Merge the PMI data with the weight data by date
df_merged = pd.merge_asof(df_weight, df_pmi, on='Date', direction='backward')

# Forward fill missing values after merge
df_merged = df_merged.ffill()

# Ensure no NaN values remain after forward fill
df_merged = df_merged.dropna()

# Scale both Weight and PMI to the range [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
df_merged[['Weight_scaled', 'PMI_scaled']] = scaler.fit_transform(df_merged[['Weight', 'Value']])

# Calculate the correlation between scaled Weight and scaled PMI
correlation = df_merged['Weight_scaled'].corr(df_merged['PMI_scaled'])
print(f"Correlation between Scaled Weight and Scaled PMI: {correlation:.2f}")

# Plot the scaled Weight and PMI
plt.figure(figsize=(12, 6))

# Plot scaled Weight
plt.plot(df_merged['Date'], df_merged['Weight_scaled'], 'b-o', label='Scaled Weight (Weekly)')

# Plot scaled PMI
plt.plot(df_merged['Date'], df_merged['PMI_scaled'], 'r--', label='Scaled PMI (Monthly)')

# Add correlation text to the plot
plt.text(df_merged['Date'].iloc[0], df_merged['Weight_scaled'].max(), 
         f'Correlation: {correlation:.2f}', 
         color='black', fontsize=12, ha='left')

# Customize the plot
plt.title('Scaled Weight (Weekly) and Scaled PMI (Monthly) - China')
plt.xlabel('Date')
plt.ylabel('Scaled Value')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
