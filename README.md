
# Apple Weight Data Analysis

This repository contains Python scripts and tools for analyzing weight data exported from Apple Health. The analysis includes data cleaning, trend analysis using polynomial regression, exponential moving average (EMA), and time-series forecasting with ARIMA. It is designed to provide insights into weight trends and predictions based on historical data.

## Features

- **Data Cleaning**: Handles duplicate dates by averaging weights and detects/removes outliers based on IQR.
- **Trend Analysis**:
  - **Polynomial Regression**: Provides a smooth trend line using quadratic polynomial fitting.
  - **Exponential Moving Average (EMA)**: Calculates a 14-day EMA to smooth short-term fluctuations.
- **Time-Series Forecasting**: Uses ARIMA for 15 and 30-day forecasts from the last data point.
- **Visualizations**: Plots weight data, trends, and forecasts with labeled values.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/blitzwolfz/Apple-Weight-Data-Analysis.git
   cd Apple-Weight-Data-Analysis
   ```

2. Set up the virtual environment and install dependencies:
   ```bash
   python -m venv myenv
   source myenv/bin/activate       # On Windows use `myenv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Usage

1. Export your Apple Health weight data as a CSV file named `weight_data.csv` and place it in the root directory.
2. Run the analysis script:
   ```bash
   python analyze_weight_data.py
   ```

The script will output visualizations showing the original weight data, trends, and ARIMA forecasts.

## Data Analysis and Visualization

- **Data Cleaning**: Automatically removes outliers and averages duplicate dates.
- **Trend Lines**:
  - A polynomial regression trend for visualizing long-term trends.
  - EMA for visualizing short-term trends.
- **Forecasting**: Provides ARIMA forecasts for the next 15 and 30 days.

## Example Visualization

![Weight Trend and Forecast](path/to/your/image.png)

## Requirements

- Python 3.x
- `matplotlib`
- `numpy`
- `pandas`
- `statsmodels`

## License

This project is licensed under the MIT License.
