import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from numpy.polynomial.polynomial import Polynomial

# Load the CSV file and parse dates
weight_df = pd.read_csv('weight_data.csv', parse_dates=['Date'])

# Sort by date in case data is out of order and handle duplicate dates by averaging weights
weight_df = weight_df.groupby('Date', as_index=False)['Weight'].mean().sort_values(by='Date')

# Detect and handle outliers (cap extreme values within 1.5 * IQR range)
q1 = weight_df['Weight'].quantile(0.25)
q3 = weight_df['Weight'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
weight_df['Is_Outlier'] = (weight_df['Weight'] < lower_bound) | (weight_df['Weight'] > upper_bound)
weight_df['Adjusted_Weight'] = np.where(weight_df['Is_Outlier'], np.nan, weight_df['Weight'])

# Filter out outliers for trend calculations
cleaned_data = weight_df.dropna(subset=['Adjusted_Weight']).copy()

# Generate a Days column for polynomial regression
cleaned_data['Days'] = (cleaned_data['Date'] - cleaned_data['Date'].min()).dt.days

# Polynomial Regression for Trend Prediction on Non-Outlier Data
X_poly = np.array(cleaned_data['Days']).reshape(-1, 1)
y_poly = cleaned_data['Adjusted_Weight']
poly_model = Polynomial.fit(X_poly.flatten(), y_poly, 2)  # Quadratic fit

# Generate the polynomial trend line for the existing data
cleaned_data['Poly_Trend'] = poly_model(cleaned_data['Days'])

# Identify above and below trend points
cleaned_data['Above_Trend'] = cleaned_data['Adjusted_Weight'] > cleaned_data['Poly_Trend']

# Calculate Exponential Moving Average (EMA) for non-outlier data
cleaned_data['EMA_14'] = cleaned_data['Adjusted_Weight'].ewm(span=14, adjust=False).mean()

# ARIMA model for Time Series Forecasting based on non-outlier data
arima_model = ARIMA(cleaned_data['Adjusted_Weight'], order=(1, 1, 1))
arima_model_fit = arima_model.fit()

# Forecast next 15 and 30 days with ARIMA
forecast_days = [15, 30]
future_forecasts = arima_model_fit.forecast(steps=max(forecast_days))
forecast_dates = pd.date_range(start=cleaned_data['Date'].max(), periods=max(forecast_days), freq='D')

# Select only 15th and 30th day forecasts
selected_forecast_dates = forecast_dates[forecast_days[0]-1:forecast_days[1]:14]  # 15th and 30th days
selected_forecasts = future_forecasts[forecast_days[0]-1:forecast_days[1]:14]

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot original weight data
plt.plot(weight_df['Date'], weight_df['Weight'], label='Daily Weight (lbs)', marker='o', linestyle='-', color='blue')

# Mark outliers with a different color
plt.scatter(weight_df.loc[weight_df['Is_Outlier'], 'Date'], weight_df.loc[weight_df['Is_Outlier'], 'Weight'], 
            color='red', label='Outliers', marker='x', s=50)

# Plot Exponential Moving Average (EMA) on non-outlier data
plt.plot(cleaned_data['Date'], cleaned_data['EMA_14'], label='14-Day EMA', color='orange', linewidth=2)

# Plot Polynomial Trend for the non-outlier data
plt.plot(cleaned_data['Date'], cleaned_data['Poly_Trend'], label='Polynomial Trend (Quadratic)', color='purple', linestyle='--', linewidth=2)

# Highlight points above and below the trend
plt.fill_between(cleaned_data['Date'], cleaned_data['Adjusted_Weight'], cleaned_data['Poly_Trend'], 
                 where=cleaned_data['Above_Trend'], color='green', alpha=0.3, label='Above Trend')
plt.fill_between(cleaned_data['Date'], cleaned_data['Adjusted_Weight'], cleaned_data['Poly_Trend'], 
                 where=~cleaned_data['Above_Trend'], color='red', alpha=0.3, label='Below Trend')

# Plot ARIMA forecast for 15 and 30 days
plt.plot(selected_forecast_dates, selected_forecasts, label='ARIMA Forecast (15 and 30 Days)', color='red', linestyle='--', marker='o')

# Label each forecasted point with its weight value for 15 and 30 days
for date, weight in zip(selected_forecast_dates, selected_forecasts):
    plt.text(date, weight, f'{weight:.1f}', color='red', fontsize=8, ha='right', va='bottom')

# Label each original weight point with its value
for i, (date, weight) in enumerate(zip(weight_df['Date'], weight_df['Weight'])):
    plt.text(date, weight, f'{weight:.1f}', color='black', fontsize=8, ha='right', va='bottom')

# Formatting the plot
plt.xlabel('Date')
plt.ylabel('Weight (lbs)')
plt.title('Weight Over Time with Polynomial Trend, EMA, and ARIMA Forecast for 15 and 30 Days')
plt.legend()
plt.grid(True)

# Format x-axis for better date readability
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()
