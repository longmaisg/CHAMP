import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import sin, cos, pi

# Step 1: Read the CSV file into a pandas DataFrame
df = pd.read_csv('yield.csv')

# Step 2: Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Step 3: Extract time components
df['Season'] = (df['Date'].dt.month % 12 // 3) + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
df['Quarter'] = df['Date'].dt.quarter
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['Day_of_Year'] = df['Date'].dt.dayofyear
df['Day_of_Month'] = df['Date'].dt.day
# df['Day_of_Week'] = df['Date'].dt.dayofweek + 1  # Convert 0-based to 1-based (Monday = 1, ..., Sunday = 7)

# Step 4: Calculate sine and cosine values for each time component
def calculate_value_sine(value, date_max):
    return sin(2 * pi * value / date_max)

def calculate_value_cosine(value, date_max):
    return cos(2 * pi * value / date_max)

# Define maximum values for each component
max_values = {
    'Season': 4,  # 4 seasons in a year
    'Quarter': 4,  # 4 quarters in a year
    'Month': 12,  # 12 months in a year
    'Week': 52,  # 52 weeks in a year
    'Day_of_Year': 365,  # 365 days in a year
    'Day_of_Month': 31,  # Maximum days in a month
    # 'Day_of_Week': 7  # 7 days in a wesek
}

# Calculate sine and cosine for each component
for component, date_max in max_values.items():
    df[f'{component}_sine'] = df[component].apply(lambda x: calculate_value_sine(x, date_max))
    df[f'{component}_cosine'] = df[component].apply(lambda x: calculate_value_cosine(x, date_max))

# Save the DataFrame with the new sine and cosine columns to a CSV file
df.to_csv('yield_with_sine_cosine.csv', index=False)

# Step 5: Create an interactive plot using plotly
fig = go.Figure()

# Add traces for each time component
components = [ 'Season', 'Quarter', 'Month', 'Week', 'Day_of_Year', 'Day_of_Month']
for component in components:
    fig.add_trace(go.Scatter(x=df['Date'], y=df[f'{component}_sine'],
                             mode='lines+markers',
                             name=f'{component} sine'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df[f'{component}_cosine'],
                             mode='lines+markers',
                             name=f'{component} cosine'))

# Update layout
fig.update_layout(
    title='Interactive Plot of Time Components Sine and Cosine',
    xaxis_title='Date',
    yaxis_title='Values',
    legend_title='Components',
    template='plotly_dark'
)

# Show the interactive plot
fig.show()
