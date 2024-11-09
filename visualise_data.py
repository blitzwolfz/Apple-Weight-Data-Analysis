import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np

# Load the CSV file and parse dates
weight_df = pd.read_csv('weight_data.csv', parse_dates=['Date'])

# Sort by date in case data is out of order
weight_df = weight_df.sort_values(by='Date')

# Calculate a 7-day moving average for smoother trend visualization
weight_df['7-Day Moving Average'] = weight_df['Weight'].rolling(window=7).mean()

# Calculate the linear trend line
X = np.array((weight_df['Date'] - weight_df['Date'].min()).dt.days).reshape(-1, 1)
y = weight_df['Weight'].values
# Fit the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
trend_line = model.predict(X)

# Plot the data
plt.figure(figsize=(12, 8))
plt.plot(weight_df['Date'], weight_df['Weight'], marker='o', linestyle='-', label='Daily Weight')
plt.plot(weight_df['Date'], weight_df['7-Day Moving Average'], color='orange', linestyle='-', linewidth=2, label='7-Day Moving Average')
plt.plot(weight_df['Date'], trend_line, color='red', linestyle='--', linewidth=2, label='Linear Trend Line')

# Highlighting specific points - Max and Min weight
max_weight = weight_df['Weight'].max()
min_weight = weight_df['Weight'].min()
max_date = weight_df.loc[weight_df['Weight'].idxmax(), 'Date']
min_date = weight_df.loc[weight_df['Weight'].idxmin(), 'Date']
plt.scatter(max_date, max_weight, color='green', label='Max Weight', zorder=5)
plt.scatter(min_date, min_weight, color='blue', label='Min Weight', zorder=5)

# Annotations for max and min weight points
plt.annotate(f'Max: {max_weight} kg', xy=(max_date, max_weight), xytext=(max_date, max_weight+1),
             arrowprops=dict(facecolor='green', shrink=0.05))
plt.annotate(f'Min: {min_weight} kg', xy=(min_date, min_weight), xytext=(min_date, min_weight-1),
             arrowprops=dict(facecolor='blue', shrink=0.05))

# Formatting the plot
plt.xlabel('Date')
plt.ylabel('Weight (kg)')
plt.title('Weight Over Time with Trend Analysis')
plt.legend()
plt.grid(True)

# Format x-axis for better date readability
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()
