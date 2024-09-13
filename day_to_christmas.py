import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
df = pd.read_csv('yield.csv', parse_dates=['Date'])

# Sort the dataframe by date
df = df.sort_values('Date')

# Define a function to calculate days since the last Christmas
def days_since_last_christmas(date):
    year = date.year
    christmas = datetime(year, 12, 25)
    if date < christmas:
        christmas = datetime(year - 1, 12, 25)
    return (date - christmas).days

# Apply the function to each date
df['Days_since_Last_Christmas'] = df['Date'].apply(days_since_last_christmas)

# Scale Weight and Days_since_Last_Christmas to the range [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
df[['Weight_scaled', 'Days_since_Last_Christmas_scaled']] = scaler.fit_transform(df[['Weight', 'Days_since_Last_Christmas']])

# Calculate the correlation between scaled Weight and scaled Days_since_Last_Christmas
correlation = df['Weight_scaled'].corr(df['Days_since_Last_Christmas_scaled'])
print(f"Correlation between Scaled Weight and Scaled Days Since Last Christmas: {correlation:.2f}")

# Plot the Weight and Days since Last Christmas
plt.figure(figsize=(12, 6))

# Plot scaled Weight
plt.plot(df['Date'], df['Weight_scaled'], 'b-o', label='Scaled Weight')

# Plot scaled Days since Last Christmas
plt.plot(df['Date'], df['Days_since_Last_Christmas_scaled'], 'r--', label='Scaled Days Since Last Christmas')

# Add correlation text to the plot
plt.text(df['Date'].iloc[0], df['Weight_scaled'].max(), 
         f'Correlation: {correlation:.2f}', 
         color='black', fontsize=12, ha='left')

# Customize the plot
plt.title('Scaled Weight and Scaled Days Since Last Christmas')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Scaled Value')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
