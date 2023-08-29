---
title: "Time Series Analysis of the Covid-19 pandemic in Mexico"
date: 2023-08-29 11:33:00 +0800
categories: [Covid-19 Time Series]
tags: [ARIMA, Time Series Analysis]
pin: true
math: true
---

In this post we will look at the time series workflow in python, looking at confirmed Covid-19 cases at the national and state level in Mexico. We will be following the [Box-Jenkins method](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method) for fitting time series models, which in this case will be [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) models.

## Background
ARIMA stands for AutoRegressive Integrated Moving Average, it consists of three separate components:
- Auto Regressive (AR): It reffers to predicting the current observation based on $p$ previous ones. So an AR(1) model is:
$x_t = \beta x_{t-1} + \varepsilon_t $. 
- Moving Average (MA): This accounts for the correlation between the current observation and the previous $q$ errors (difference between predicted values with an AR model and the actual data). 
- Integrated (I): This is the number of times we take the difference of the series (i.e substracting the previous observation to the current one). It's often done to work with a stationary series. Ex.: $x_t-x_{t-1} = \beta(x_{t-2}-x_{t-3}) + \varepsilon_t$.

ARIMA models requires a few assumptions on the data:
- The time series is stationary, this means that the the mean and variance of the distribution from which the data is generated must stay cconstant in time. 
- This implies that they aren't auto-correlated and have equal variance (similar to the linear regression assumption).



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Exploratory Data Analysis
The data of daily cases is made available by the Ministry of Health and the Council for Science and Technology ([Conacyt](https://datos.covid-19.conacyt.mx/)). The dataframe was downloaded and stored on github; its columns represent days and its rows are Mexican states. 


```python
daily_cases = pd.read_csv("https://raw.githubusercontent.com/abraham-mv/Covid-19-in-Mexico-time-series/main/Casos_Diarios_Estado_Nacional_Confirmados_20220109.csv")
daily_cases.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cve_ent</th>
      <th>poblacion</th>
      <th>nombre</th>
      <th>2020-02-26</th>
      <th>2020-02-27</th>
      <th>2020-02-28</th>
      <th>2020-02-29</th>
      <th>2020-03-01</th>
      <th>2020-03-02</th>
      <th>2020-03-03</th>
      <th>...</th>
      <th>2021-12-31</th>
      <th>2022-01-01</th>
      <th>2022-01-02</th>
      <th>2022-01-03</th>
      <th>2022-01-04</th>
      <th>2022-01-05</th>
      <th>2022-01-06</th>
      <th>2022-01-07</th>
      <th>2022-01-08</th>
      <th>2022-01-09</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>1380011</td>
      <td>TLAXCALA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>7</td>
      <td>15</td>
      <td>46</td>
      <td>79</td>
      <td>66</td>
      <td>103</td>
      <td>67</td>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>8539862</td>
      <td>VERACRUZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>83</td>
      <td>28</td>
      <td>68</td>
      <td>215</td>
      <td>238</td>
      <td>255</td>
      <td>252</td>
      <td>169</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>2259098</td>
      <td>YUCATAN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>295</td>
      <td>49</td>
      <td>73</td>
      <td>629</td>
      <td>711</td>
      <td>784</td>
      <td>715</td>
      <td>574</td>
      <td>52</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>1666426</td>
      <td>ZACATECAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>218</td>
      <td>58</td>
      <td>76</td>
      <td>465</td>
      <td>609</td>
      <td>697</td>
      <td>711</td>
      <td>410</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>127792286</td>
      <td>Nacional</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>9</td>
      <td>...</td>
      <td>9453</td>
      <td>2969</td>
      <td>4668</td>
      <td>19513</td>
      <td>24564</td>
      <td>26464</td>
      <td>23679</td>
      <td>15790</td>
      <td>4284</td>
      <td>67</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 687 columns</p>
</div>



We'll mainly focus on the national level first, let's take a look at our time series.


```python
import matplotlib.pyplot as plt
nation = daily_cases.iloc[-1,:].drop(["cve_ent", "poblacion", "nombre"], axis=0)
nation.name = "National"
nation.index = pd.to_datetime(nation.index, infer_datetime_format=True)
nation.plot()
plt.grid()
```


    
![png](/img/covid-19-time-series/output_5_0.png)
    


We can see three clear peaks in daily cases, corresponding to the first 3 waves of the pandemic in the country. The time series shows highly perdiodic behaviour let's decompose the first 150 observations into its trend seasonal and residual parts. 


```python
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(nation[:100]).plot()
plt.show()
```


    
![png](/img/covid-19-time-series/output_7_0.png)
    


There's a clear upward trend at the beginning of the pandemic. Additionally, strong seasonality is evident in the data, and the residuals also exhibit some periodicity. This seasonality could potentially be attributed to a shortage of staff members collecting patient information during weekends, rather than indicating an actual drop in cases. A common approach for addressing this type of time series pattern is to apply smoothing using a moving average technique. This involves calculating the average of the last $n$ observations.

Given that the decline in cases seems to occur on a weekly basis, it would be appropriate to use a smoothing window of 7 days or any multiple of 7.


```python
X_sma_7 = nation.rolling(window=7).mean().dropna()
X_sma_14 = nation.rolling(window=14).mean().dropna()
X_sma_21 = nation.rolling(window=21).mean().dropna()
plt.figure(figsize=(7,5),dpi=100)
nation.plot(label = 'Daily cases', alpha=0.8)
X_sma_7.plot(color = 'black',label = '7 day average')
X_sma_14.plot(color = 'darkgreen', label = '14 day average')
X_sma_21.plot(color = 'red', label = '21 day average')
plt.grid()
plt.legend()
plt.show()
```


    
![png](/img/covid-19-time-series/output_9_0.png)
    


Important to note that as we increase the window the series becomes much smoother.

## Data transformations and model selection

The next step in the time series workflow is model selection. When utilizing ARIMA models a common practice is to determine the coefficients using plots of the Autocorrelation and Partial Autocorrelation Functions ([ACF](https://en.wikipedia.org/wiki/Autocorrelation) & [PACF](https://en.wikipedia.org/wiki/Partial_autocorrelation_function)).The former represents the correlation between the series and its past values, while the latter signifies the autocorrelation among the residuals of an auto-regressive model. These plots are also highly beneficial for identifying patterns in the data and ascertaining whether the series is stationary.

For the purposes of this exercise, we will truncate the series for training starting from July 1st, 2021. This date was chosen as it marked a period when cases were experiencing an increase, and forecasting future daily cases held significant general interest.


```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

train = X_sma_7[X_sma_7.index < pd.to_datetime('2021-07-01')]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True)
axes = axes.ravel() 
plot_acf(train, ax=axes[0])
plot_pacf(train, ax=axes[1])
plt.show()
```


    
![png](/img/covid-19-time-series/output_11_0.png)
    


Regarding the smoothed series, the Autocorrelation Function (ACF) displays a decreasing trend, while the Partial Autocorrelation Function (PACF) exhibits significant peaks at lags 1 and 2. This observation implies that the series's dependence is primarily on the preceding 2 days. Consequently, the most suitable model would be an AR(2).

To induce stationarity, it's common to take one or two differences. Other transformations, such as applying the natural logarithm or taking the square root of the series, can also be employed to mitigate the impact of extreme values. Let's proceed by examining this transformed series.


```python
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 6), sharex=True)
axes = axes.ravel() 
train.diff().plot(ax = axes[0], title="One difference", grid=True)
np.log(train).diff().plot(ax = axes[1], title="Log transform & 1 difference", grid=True)
np.sqrt(train).diff().plot(ax = axes[2], title="Square root & 1 difference", grid=True)
train.diff().diff().plot(ax = axes[3], title="Two differences", grid=True)
np.log(train).diff().diff().plot(ax = axes[4], title="Log transform & 2 differences", grid=True)
np.sqrt(train).diff().diff().plot(ax = axes[5], title="Square root & 2 differences", grid=True)
plt.show()
```


    
![png](/img/covid-19-time-series/output_13_0.png)
    


It's apparent that with just one difference, the series's mean continues to exhibit temporal variation. This issue can be addressed by taking an additional difference. Nevertheless, despite applying log or square root transformations, the variance of the series remains variable. Upon examining these graphs, the most viable approach might be to combine the log transformation with a second-order integration (2 differences). 


```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True)
axes = axes.ravel()
plot_acf(np.log(train).diff().diff().dropna(), ax=axes[0])
plot_pacf(np.log(train).diff().diff().dropna(), ax=axes[1])
plt.show()
```


    
![png](/img/covid-19-time-series/output_15_0.png)
    


Looking at the ACF and PACF plots for this transformed series, we may be looking at an ARMA process here. 

## Model Fitting and validation
To determine the exact order of the ARIMA model we use the `auto_arima` function from the `pmdarima` package. This function will quickly fit models and compute the AIC, the model with the highest AIC is selected.


```python
import pmdarima
arima_model = pmdarima.arima.auto_arima(np.log(train), d=2)
arima_model.summary()
```




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>    <td>485</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 2, 1)</td> <th>  Log Likelihood     </th>  <td>930.381</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Tue, 29 Aug 2023</td> <th>  AIC                </th> <td>-1854.762</td>
</tr>
<tr>
  <th>Time:</th>                <td>15:14:32</td>     <th>  BIC                </th> <td>-1842.222</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>-1849.834</td>
</tr>
<tr>
  <th></th>                      <td> - 485</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>  <td>    0.3424</td> <td>    0.034</td> <td>   10.181</td> <td> 0.000</td> <td>    0.276</td> <td>    0.408</td>
</tr>
<tr>
  <th>ma.L1</th>  <td>   -0.8842</td> <td>    0.024</td> <td>  -36.850</td> <td> 0.000</td> <td>   -0.931</td> <td>   -0.837</td>
</tr>
<tr>
  <th>sigma2</th> <td>    0.0012</td> <td> 3.64e-05</td> <td>   34.120</td> <td> 0.000</td> <td>    0.001</td> <td>    0.001</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.22</td> <th>  Jarque-Bera (JB):  </th> <td>1859.44</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.64</td> <th>  Prob(JB):          </th>  <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.77</td> <th>  Skew:              </th>  <td>0.25</td>  
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.09</td> <th>  Kurtosis:          </th>  <td>12.60</td> 
</tr>
</table>



The function suggest an ARMA(1,1) on the second order integrated and log-transformed series, without any seasonality component. It's important to note the following from the results:  
- The coefficients for the AR and MA components are both statistically significant. 
- The Ljung-Box test is showing a p-value greater than 0.05, this means that we fail to reject the null-hypothesis that the residuals are independent (although we should expect something closer to 0.99 with a better model).
- The heteroskedasticity test is showing a p-value greater than 0.05, therefore we can't reject the null-hypothesis that the error variances are independent (again we would expect to have a higher p-value with a better model).

We can perform many tests, but nothing beats looking the graph and check if the model assumptions seem to be fulfilled.


```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
axes = axes.ravel()
pd.Series(arima_model.resid(), index=train.index).plot(color="black", ax=axes[0], title="Residuals")
plot_acf(arima_model.resid(), title="ACF of Residuals", ax=axes[1])
plt.show()
```


    
![png](/img/covid-19-time-series/output_19_0.png)
    


From this graph, it is evident that while the mean seems to be independent of time, there are periods where the variance is significantly higher, indicating the presence of heteroskedasticity. In an ideal scenario, a perfect model would yield Gaussian noise behavior. The ACF plot demonstrates a few significant peaks, notably at lags 1 and 8. This outcome is suboptimal and further confirms the likelihood of some correlation within the residuals.

While this model isn't perfect by any means, it's probably as good as it gets for this dataset.

## Forecasting
We will use the model to predict 30 days in the future.


```python
test = X_sma_7[X_sma_7.index >= pd.to_datetime('2021-07-01')][:35]
prediction, confint = arima_model.predict(n_periods=35, return_conf_int=True)
prediction
```




    array([8.52809307, 8.55922593, 8.59075826, 8.62242736, 8.65414329,
           8.68587525, 8.7176127 , 8.74935204, 8.78109201, 8.81283221,
           8.84457248, 8.87631278, 8.90805309, 8.9397934 , 8.97153371,
           9.00327402, 9.03501433, 9.06675464, 9.09849496, 9.13023527,
           9.16197558, 9.19371589, 9.2254562 , 9.25719652, 9.28893683,
           9.32067714, 9.35241745, 9.38415776, 9.41589808, 9.44763839,
           9.4793787 , 9.51111901, 9.54285932, 9.57459964, 9.60633995])




```python
cf = pd.DataFrame(confint)
prediction_series = pd.Series(prediction,index=test.index)
fig, ax = plt.subplots(1, 1, figsize=(8,5))
np.log(train[-180:]).plot(ax=ax, label="train")
np.log(test).plot(ax=ax, label="test")
ax.plot(prediction_series, label="forecast")
ax.legend(loc="upper left")
ax.fill_between(prediction_series.index,
                cf[0],
                cf[1],color='green',alpha=.3, label="CI")
ax.set_title("Log-transformed daily cases with 35 day forecast", fontsize=14)
ax.grid()
```


    
![png](/img/covid-19-time-series/output_23_0.png)
    


As we can see the model is succesfully able to capture the upward trend of the series, but a major limitation of these models is their unability to determine when a peak is coming. Next step is to measure the performance of the model on the testing set, we will do this using the root mean square error, mean absolute error and mean absolute percentage error from the `sklearn` package.


```python
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

metrics = pd.DataFrame({"MAE": mean_absolute_error(test, np.exp(prediction)), 
                        "RMSE": np.sqrt(mean_squared_error(test, np.exp(prediction))), 
                        "MAPE": mean_absolute_percentage_error(test, np.exp(prediction)), 
                        "R2": r2_score(test, np.exp(prediction))}, 
                      index=range(1))
metrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>MAPE</th>
      <th>R2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2634.686582</td>
      <td>3094.601593</td>
      <td>0.19656</td>
      <td>0.4955</td>
    </tr>
  </tbody>
</table>
</div>



## Conclusion
In this post we looked at how to approach time series problems using the Box-Jenkins method and ARIMA models. In my opinion is clear that for this type of data these models might not be the best match, given that there are a lot of external variables, population constrains and health measures taken by the authorities that affect the behavior of the series.
