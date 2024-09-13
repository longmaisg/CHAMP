import pandas as pd
import matplotlib.pyplot as plt

# Initialize an empty DataFrame with the appropriate column names
df = pd.DataFrame(columns=['Date', 'Yield', 'Weight'])

# Open the file and read it line by line
with open('yield.txt', 'r') as file:
    for line in file:
        # Split each line by comma
        try:
            date_str, yield1, yield2 = line.strip().split(',')
            
            # Create a temporary DataFrame for this line
            temp_df = pd.DataFrame([[date_str, float(yield1), float(yield2)]], columns=['Date', 'Yield', 'Weight'])
            
            # Append it to the main DataFrame
            df = pd.concat([df, temp_df], ignore_index=True)
        
        except ValueError as e:
            print(f"Skipping line due to error: {line} ({e})")

# Convert 'Date' column to datetime format (optional)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Save the DataFrame to a CSV file
df.to_csv('yield.csv', index=False)

# Plot the Yield and Weight columns
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Yield'], label='Yield', marker='o')
plt.plot(df['Date'], df['Weight'], label='Weight', marker='x')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Yield and Weight Over Time')
plt.legend()

# Display the plot
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()