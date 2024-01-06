# Import packages
```python
!pip install pmdarima

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from datetime import datetime
from scipy.optimize import curve_fit
```

# SARIMA Forecast w/ Threshold-Based Decision Model
```python
# Time series plot
# Variance seem constant over time, so no need to log outcome but could try later to see if it improves results

plt.figure(figsize=(10, 6))
sns.lineplot(data=ramona_df, x='time', y='HSI', hue='year')
plt.title('Annual seasonal behavior, slight upward trend overall')
plt.suptitle('Time Series Plot of HSI')
plt.xlabel('Time')
plt.ylabel('HSI')
plt.grid(True)
plt.legend(title='Year')
plt.show()
```
![Time Series Plot of Heat Stress Index](https://github.com/haschuele/EnergyDM/blob/main/Time%20Series%20Plot%20of%20HSI.png)

```python
# Plot ACF Annually

# Oscillating pattern points to seasonality

# Compute the ACF
acf = sm.tsa.acf(ramona_train['HSI'], nlags=8760)  # nlags specifies the number of lags to include in the ACF plot

# Create the ACF plot
plt.figure(figsize=(10, 4))
plt.bar(range(len(acf)), acf)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('Annual Autocorrelation Function (ACF)')
plt.grid(True)
plt.show()
```
![ACF Plot](https://github.com/haschuele/EnergyDM/blob/main/ACF%20Plot.png)

```python
# Fit SARIMA (i.e. with seasonal component)

# Define the 'HSI' variable as a time series
hsi_series_ramona = pd.Series(ramona_train['HSI'].values, index=ramona_train['time'])

# Fit ARIMA models and select the best model using BIC
best_bic_ramona = np.inf
best_order_ramona = None
best_seasonal_order_ramona = None


# NOTES
# Set d = range(2) because of previous arima results and generally the standard not to difference more than once. Need to to ADF test for stationarity?
# May need higher P and Q values if pronounced seasonality and if ACF/PACF show spikes at lags that are multiple of seasonal period
# Test for residuals centered around 0? If so, may need higher pdq/PDQ values
# Setting pdq and PDQ values low so code will run in a decent amount of time

# Loop through possible ARIMA orders
# 64 iterations took 13 min
for p in range(2):  # Adjust the range as needed
    for d in range(2):  # Adjust the range as needed
        for q in range(2):  # Adjust the range as needed
            for P in range(2):  # Adjust the range as needed
                for D in range(2):  # Adjust the range as needed
                    for Q in range(2):  # Adjust the range as needed
                        try:
                            model = SARIMAX(hsi_series_ramona, order=(p, d, q), seasonal_order=(P, D, Q, 12))  # Assumes 12-month seasonal
                            results = model.fit()
                            bic = results.bic
                            if bic < best_bic_ramona:
                                best_bic_ramona = bic
                                best_order_ramona = (p, d, q)
                                best_seasonal_order_ramona = (P, D, Q, 12)
                        except:
                            continue

# Fit the best SARIMA model
best_model_ramona = SARIMAX(hsi_series_ramona, order=best_order_ramona, seasonal_order=best_seasonal_order_ramona)
results_ramona = best_model_ramona.fit()

# Print the best ARIMA order and BIC
print(f"Best Ramona SARIMA Order (p, d, q): {best_order_ramona}")
print(f"Best Ramona Seasonal Order (P, D, Q, s): {best_seasonal_order_ramona}")
print(f"Best BIC: {best_bic_ramona}")

# THIS IS A BAD MODEL, NOT ENOUGH COMPUTE POWER TO TRY MORE
# Best Ramona SARIMA Order (p, d, q): (1, 0, 0)
# Best Ramona Seasonal Order (P, D, Q, s): (1, 0, 1, 12)
# Could try data reduction techniques so at e.g. daily max instead of hourly

# Summary of the model
print(results_ramona.summary())
```
![SARIMA Results](https://github.com/haschuele/EnergyDM/blob/main/SARIMA%20Results.png)

# Quantile Forecast with NOAA Anomaly Detection
```python
# Plot quantile forecasts with orange dots when NOAA forecasted HSI > p70 HSI

# Create a figure
plt.figure(figsize=(10, 6))

# Plot p20 and p70 as blue/red lines
plt.plot(mariposa_quant_forecast.index, mariposa_quant_forecast['p10'], label='p10', color='blue', linewidth=0.2)
plt.plot(mariposa_quant_forecast.index, mariposa_quant_forecast['p90'], label='p90', color='red', linewidth=0.2)

# Black line for p50 (median)
plt.plot(mariposa_quant_forecast.index, mariposa_quant_forecast['p50'], label='p50', color='black', linewidth=0.2)

# Merge the two DataFrames on the common index
merged_df = mariposa_quant_forecast.merge(mariposa_noaa_anom, left_index=True, right_index=True, suffixes=('_forecast', '_noaa'))
merged_df

# Overlay orange dots where HSI > p70
condition = merged_df['HSI'] > merged_df['p90']
plt.scatter(merged_df.index[condition], merged_df['HSI'][condition], c='orange', label='HSI > p90')

# Add labels and legend
first_date = str(mariposa_quant_forecast.index[0])
last_date = str(mariposa_quant_forecast.index[-1])
plt.xlabel('Time')
plt.ylabel('HSI')
plt.title(f'{first_date} through {last_date}')
plt.suptitle('Line Plot of p10, p50, and p90 Quantiles with HSI > p90 as Orange Dots')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
```
![Quantile Anomaly Detection](https://github.com/haschuele/EnergyDM/blob/main/Quantile%20Anomaly%20Detection.png)

# Time Series Decomposition with Percentile Anomaly Detection
```python
# Define sine function
# TO DO compare this to np.sin()
# amplitude (A), angular frequency (omega), phase (phi), and the vertical shift (c).

# Amplitude (A): The amplitude parameter (A) determines the height of the peaks and the depth of the troughs in the seasonal cycle. It reflects the extent to which the data values deviate from the mean.
# Frequency (Omega): The angular frequency parameter (omega) controls how many oscillations occur within a given time interval. It defines the length of the seasonal cycle, with higher values leading to shorter cycles and more oscillations.
# Phase (Phi): The phase parameter (phi) indicates the horizontal shift or offset of the seasonal cycle. It determines where the cycle begins in time.
# Vertical Shift (C): The vertical shift parameter (c) represents a constant value that is added to the sine function. It sets the baseline or average level of the seasonal cycle.

# OBSERVATION Sine is unable to capture seasonality of longer periods of time (e.g. a few months) because it's too complex

def sine_func(x, A, omega, phi, c):
    return A * np.sin(omega * x + phi) + c

# Fit sine function to data
x_data = np.arange(len(ramona_df))  # Index or time values
y_data = ramona_df["HSI"].values  # Your time series data

# params, params_covariance = curve_fit(sine_func, x_data, y_data)
params, params_covariance = curve_fit(sine_func, x_data, y_data, p0=[0.75, 0.3, 0, 0]) # With initial parameter estimates
A, omega, phi, c = params  # Extract the fitted parameters


# NOTES
# The fitted seasonal cycle should exhibit similar characteristics to the original data, such as the frequency and amplitude of the seasonal patterns.
# It should provide a smooth and continuous representation of the seasonal component, helping to remove noise and emphasize the underlying trends.

# Generate seasonal cycle using fitted parameters
seasonal_cycle = sine_func(x_data, A, omega, phi, c)

# Plot original and fitted series
plt.figure(figsize=(12, 6))
plt.plot(ramona_df['time'].values, y_data, label='Original Data', color='blue')
plt.plot(ramona_df['time'].values, seasonal_cycle, label='Fitted Seasonal Cycle', color='red')
plt.xlabel('Time')  # Use 'Time' as the x-axis label
plt.ylabel('Value')
plt.title('Fitted Seasonal Cycle using Sine Function')
plt.legend()
plt.grid(True)
plt.show()
```
![Sinusoidal Seasonal Fit](https://github.com/haschuele/EnergyDM/blob/main/Sinusoidal%20Fit.png)
