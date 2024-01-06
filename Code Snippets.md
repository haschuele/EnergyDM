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
