import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('yield.csv', parse_dates=['Date'])

# Sort the dataframe by date
df = df.sort_values('Date')

# Calculate the 4-week moving average of 'Weight'
df['4_week_avg_weight'] = df['Weight'].rolling(window=4).mean()

# Shift the 'Weight' column by 4 weeks to represent the weight 4 weeks into the future
df['weight_in_4_weeks'] = df['Weight'].shift(-4)

# Drop any rows where '4_week_avg_weight' or 'weight_in_4_weeks' is NaN
df = df.dropna()

# Calculate the correlation between 4-week moving average and 'weight_in_4_weeks'
correlation = df['4_week_avg_weight'].corr(df['weight_in_4_weeks'])
print(f"Correlation between 4-week average weight and weight in next 4 weeks: {correlation}")

# Plot the relationship between 4-week average weight and weight in next 4 weeks
plt.figure(figsize=(8, 6))
plt.scatter(df['4_week_avg_weight'], df['weight_in_4_weeks'], label=f'Correlation: {correlation:.2f}')

# Adding the baseline (1:1 line)
min_val = min(df['4_week_avg_weight'].min(), df['weight_in_4_weeks'].min())
max_val = max(df['4_week_avg_weight'].max(), df['weight_in_4_weeks'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Baseline (1:1)')

# Customize the plot
plt.title('4-week Average Weight vs Weight in Next 4 Weeks')
plt.xlabel('4-week Average Weight')
plt.ylabel('Weight in 4 Weeks')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
