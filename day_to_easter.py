import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the CSV file
df = pd.read_csv('yield.csv', parse_dates=['Date'])

# Sort the dataframe by date
df = df.sort_values('Date')

# Function to calculate the date of Easter for a given year
def easter_date(year):
    "Returns the date of Easter for a given year."
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day)

# Function to calculate days since the last Easter
def days_since_last_easter(date):
    year = date.year
    easter = easter_date(year)
    if date < easter:
        easter = easter_date(year - 1)
    return (date - easter).days

# Apply the function to each date
df['Days_since_Last_Easter'] = df['Date'].apply(days_since_last_easter)

# Scale Weight and Days_since_Last_Easter to the range [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
df[['Weight_scaled', 'Days_since_Last_Easter_scaled']] = scaler.fit_transform(df[['Weight', 'Days_since_Last_Easter']])

# Calculate the correlation between scaled Weight and scaled Days_since_Last_Easter
correlation = df['Weight_scaled'].corr(df['Days_since_Last_Easter_scaled'])
print(f"Correlation between Scaled Weight and Scaled Days Since Last Easter: {correlation:.2f}")

# Plot the Weight and Days since Last Easter
plt.figure(figsize=(12, 6))

# Plot scaled Weight
plt.plot(df['Date'], df['Weight_scaled'], 'b-o', label='Scaled Weight')

# Plot scaled Days since Last Easter
plt.plot(df['Date'], df['Days_since_Last_Easter_scaled'], 'r--', label='Scaled Days Since Last Easter')

# Add correlation text to the plot
plt.text(df['Date'].iloc[0], df['Weight_scaled'].max(), 
         f'Correlation: {correlation:.2f}', 
         color='black', fontsize=12, ha='left')

# Customize the plot
plt.title('Scaled Weight and Scaled Days Since Last Easter')
plt.xlabel('Date')
plt.ylabel('Scaled Value')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
