<a class="anchor" id="0"></a>

# **ARIMA Model for Time Series Forecasting**



Hello friends, 


In this notebook, I will discuss **ARIMA Model for time series forecasting**. ARIMA model is used to forecast a time series using the series past values. In this notebokk, we build an **optimal ARIMA model** and extend it to **Seasonal ARIMA (SARIMA)** and **SARIMAX models**. We will also see how to build autoarima models in python.


So, let's get started.



<a class="anchor" id="0.1"></a>
# **Table of Contents** 


1.	[Introduction to Time Series Forecasting](#1)
2.	[Introduction to ARIMA Models](#2)
3.	[The meaning of p, d and q in ARIMA model](#3)
4.	[AR and MA models](#4)
5.	[How to find the order of differencing (d) in ARIMA model](#5)
6.	[How to find the order of the AR term (p)](#6)
7.	[How to find the order of the MA term (q)](#7)
8.	[How to handle if a time series is slightly under or over differenced](#8)
9.	[How to build the ARIMA Model](#9)
10.	[Find the optimal ARIMA model using Out-of-Time Cross validation](#10)
11.	[Accuracy Metrics for Time Series Forecast](#11)
12.	[Auto Arima Forecasting in Python](#12)
13.	[How to interpret the residual plots in ARIMA model](#13)
14.	[SARIMA model in python](#14)
15.	[SARIMAX model with exogeneous variables](#15)
16.	[References](#16)


# **1. Introduction to Time Series Forecasting** <a class="anchor" id="1"></a>

[Table of Contents](#0.1)


- A **Time Series** is defined as a series of data points recorded at different time intervals. The time order can be daily, monthly, or even yearly.

- Time Series forecasting is the process of using a statistical model to predict future values of a time series based on past results.

- Forecasting is the step where we want to predict the future values the series is going to take. Forecasting a time series is often of tremendous commercial value.

#### **Forecasting a time series can be broadly divided into two types.**

- If we use only the previous values of the time series to predict its future values, it is called **Univariate Time Series Forecasting.**

- If we use predictors other than the series (like exogenous variables) to forecast it is called **Multi Variate Time Series Forecasting.**

- This notebook focuses on a particular type of forecasting method called **ARIMA modeling.**


# **2. Introduction to ARIMA Models** <a class="anchor" id="2"></a>

[Table of Contents](#0.1)


- **ARIMA** stands for **Autoregressive Integrated Moving Average Model**. It belongs to a class of models that explains a given time series based on its own past values -i.e.- its own lags and the lagged forecast errors. The equation can be used to forecast future values. Any ‘non-seasonal’ time series that exhibits patterns and is not a random white noise can be modeled with ARIMA models.


- So, **ARIMA**, short for **AutoRegressive Integrated Moving Average**, is a forecasting algorithm based on the idea that the information in the past values of the time series can alone be used to predict the future values.


- **ARIMA Models** are specified by three order parameters: (p, d, q), 

   where,

   - p is the order of the AR term

   - q is the order of the MA term

   - d is the number of differencing required to make the time series stationary


- **AR(p) Autoregression** – a regression model that utilizes the dependent relationship between a current observation and observations over a previous period. An auto regressive (AR(p)) component refers to the use of past values in the regression equation for the time series.


- **I(d) Integration** – uses differencing of observations (subtracting an observation from observation at the previous time step) in order to make the time series stationary. Differencing involves the subtraction of the current values of a series with its previous values d number of times.


- **MA(q) Moving Average** – a model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations. A moving average component depicts the error of the model as a combination of previous error terms. The order q represents the number of terms to be included in the model.


## **Types of ARIMA Model**


- **ARIMA** : Non-seasonal Autoregressive Integrated Moving Averages
- **SARIMA** : Seasonal ARIMA
- **SARIMAX** : Seasonal ARIMA with exogenous variables



If a time series, has seasonal patterns, then we need to add seasonal terms and it becomes SARIMA, short for **Seasonal ARIMA**.


# **3. The meaning of p, d and q in ARIMA model** <a class="anchor" id="3"></a>

[Table of Contents](#0.1)


## **3.1 The meaning of p**


- `p` is the order of the **Auto Regressive (AR)** term. It refers to the number of lags of Y to be used as predictors.



## **3.2 The meaning of d**


- The term **Auto Regressive**’ in ARIMA means it is a linear regression model that uses its own lags as predictors. Linear regression models, as we know, work best when the predictors are not correlated and are independent of each other. So we need to make the time series stationary.


- The most common approach to make the series stationary is to difference it. That is, subtract the previous value from the current value. Sometimes, depending on the complexity of the series, more than one differencing may be needed.


- The value of d, therefore, is the minimum number of differencing needed to make the series stationary. If the time series is already stationary, then d = 0.



## **3.3 The meaning of q**


- **q** is the order of the **Moving Average (MA)** term. It refers to the number of lagged forecast errors that should go into the ARIMA Model.


# **4. AR and MA models** <a class="anchor" id="4"></a>

[Table of Contents](#0.1)



## **4.1 AR model**


- An **Auto Regressive (AR) model** is one where Yt depends only on its own lags. 

- That is, Yt is a function of the `lags of Yt`. It is depicted by the following equation -


![AR Model](https://www.machinelearningplus.com/wp-content/uploads/2019/02/Equation-1-min.png?ezimgfmt=ng:webp/ngcb1)


image source : https://www.machinelearningplus.com/wp-content/uploads/2019/02/Equation-1-min.png?ezimgfmt=ng:webp/ngcb1


where, 

- $Y{t-1}$ is the lag1 of the series, 

- $\beta1$ is the coefficient of lag1 that the model estimates, and 

- $\alpha$ is the intercept term, also estimated by the model.


## **4.2 MA model**


- Likewise a **Moving Average (MA) model** is one where Yt depends only on the lagged forecast errors. It is depicted by the following equation -


![MA Model](https://www.machinelearningplus.com/wp-content/uploads/2019/02/Equation-2-min.png?ezimgfmt=ng:webp/ngcb1)


image source : https://www.machinelearningplus.com/wp-content/uploads/2019/02/Equation-2-min.png?ezimgfmt=ng:webp/ngcb1



where the error terms are the errors of the autoregressive models of the respective lags. 


The errors Et and E(t-1) are the errors from the following equations :


![Error Terms of the AR Model](https://www.machinelearningplus.com/wp-content/uploads/2019/02/Equation-3-min.png?ezimgfmt=ng:webp/ngcb1)


image source : https://www.machinelearningplus.com/wp-content/uploads/2019/02/Equation-3-min.png?ezimgfmt=ng:webp/ngcb1



Thus, we have discussed AR and MA Models respectively.


## **4.3 ARIMA model**


- An ARIMA model is one where the time series was differenced at least once to make it stationary and we combine the AR and the MA terms. So the equation of an ARIMA model becomes :


![ARIMA Model](https://www.machinelearningplus.com/wp-content/uploads/2019/02/Equation-4-min-865x77.png?ezimgfmt=ng:webp/ngcb1)


image source : https://www.machinelearningplus.com/wp-content/uploads/2019/02/Equation-4-min-865x77.png?ezimgfmt=ng:webp/ngcb1



### **ARIMA model in words**:


Predicted Yt = Constant + Linear combination Lags of Y (upto p lags) + Linear Combination of Lagged forecast errors (upto q lags)


# **5. How to find the order of differencing (d) in ARIMA model**  <a class="anchor" id="5"></a>

[Table of Contents](#0.1)


- As stated earlier, the purpose of differencing is to make the time series stationary. But we should be careful to not over-difference the series. An over differenced series may still be stationary, which in turn will affect the model parameters.


- So we should determine the right order of differencing. The right order of differencing is the minimum differencing required to get a near-stationary series which roams around a defined mean and the ACF plot reaches to zero fairly quick.


- If the autocorrelations are positive for many number of lags (10 or more), then the series needs further differencing. On the other hand, if the lag 1 autocorrelation itself is too negative, then the series is probably over-differenced.


- If we can’t really decide between two orders of differencing, then we go with the order that gives the least standard deviation in the differenced series.


- Now, we will explain these concepts with the help of an example as follows:-


- First, I will check if the series is stationary using the **Augmented Dickey Fuller test (ADF Test)**, from the statsmodels package. The reason being is that we need differencing only if the series is non-stationary. Else, no differencing is needed, that is, d=0.


- The null hypothesis (Ho) of the ADF test is that the time series is non-stationary. So, if the p-value of the test is less than the significance level (0.05) then we reject the null hypothesis and infer that the time series is indeed stationary.


- So, in our case, if P Value > 0.05 we go ahead with finding the order of differencing.


## **Routine set up**


## **Import data**


- Now, we will continue with our example.


- Since p-value(1.00) is greater than the significance level(0.05), let’s difference the series and see how the autocorrelation plot looks like.


- For the above data, we can see that the time series reaches stationarity with two orders of differencing.


# **6. How to find the order of the AR term (p)** <a class="anchor" id="6"></a>

[Table of Contents](#0.1)


- The next step is to identify if the model needs any AR terms. We will find out the required number of AR terms by inspecting the **Partial Autocorrelation (PACF) plot**.


- **Partial autocorrelation** can be imagined as the correlation between the series and its lag, after excluding the contributions from the intermediate lags. So, PACF sort of conveys the pure correlation between a lag and the series. This way, we will know if that lag is needed in the AR term or not.


- Partial autocorrelation of lag (k) of a series is the coefficient of that lag in the autoregression equation of Y.


$$Yt = \alpha0 + \alpha1 Y{t-1} + \alpha2 Y{t-2} + \alpha3 Y{t-3}$$


- That is, suppose, if Y_t is the current series and Y_t-1 is the lag 1 of Y, then the partial autocorrelation of lag 3 (Y_t-3) is the coefficient $\alpha_3$ of Y_t-3 in the above equation.


- Now, we should find the number of AR terms. Any autocorrelation in a stationarized series can be rectified by adding enough AR terms. So, we initially take the order of AR term to be equal to as many lags that crosses the significance limit in the PACF plot.


- We can see that the PACF lag 1 is quite significant since it is well above the significance line. So, we will fix the value of p as 1.


# **7. How to find the order of the MA term (q)** <a class="anchor" id="7"></a>

[Table of Contents](#0.1)


- Just like how we looked at the PACF plot for the number of AR terms, we will look at the ACF plot for the number of MA terms. An MA term is technically, the error of the lagged forecast.


- The ACF tells how many MA terms are required to remove any autocorrelation in the stationarized series.


- Let’s see the autocorrelation plot of the differenced series.


- We can see that couple of lags are well above the significance line. So, we will fix q as 2. If there is any doubt, we will go with the simpler model that sufficiently explains the Y.


# **8. How to handle if a time series is slightly under or over differenced** <a class="anchor" id="8"></a>

[Table of Contents](#0.1)


- It may happen that the time series is slightly under differenced. Differencing it one more time makes it slightly over-differenced. 


- If the series is slightly under differenced, adding one or more additional AR terms usually makes it up. Likewise, if it is slightly over-differenced, we will try adding an additional MA term.


# **9. How to build the ARIMA Model** <a class="anchor" id="9"></a>


[Table of Contents](#0.1)


Now, we have determined the values of p, d and q. We have everything needed to fit the ARIMA model. We will use the ARIMA() implementation in the statsmodels package.


- The model summary provides lot of information. The table in the middle is the coefficients table where the values under ‘coef’ are the weights of the respective terms.

- The coefficient of the MA2 term is close to zero and the P-Value in ‘P>|z|’ column is highly insignificant. It should ideally be less than 0.05 for the respective X to be significant.

- So, we will rebuild the model without the MA2 term.


- The model AIC has slightly reduced, which is good. The p-values of the AR1 and MA1 terms have improved and are highly significant (<< 0.05).

- Let’s plot the residuals to ensure there are no patterns (that is, look for constant mean and variance).


- The residual errors seem fine with near zero mean and uniform variance. Let’s plot the actuals against the fitted values using **plot_predict()**.


- When we set dynamic=False the in-sample lagged values are used for prediction. That is, the model gets trained up until the previous value to make the next prediction. This can make the fitted forecast and actuals look artificially good.

- So, we seem to have a decent ARIMA model. But, we can’t say that this is the best ARIMA model because we haven’t actually forecasted into the future and compared the forecast with the actual performance.

- So, the real validation we need now is the Out-of-Time cross-validation, discussed next.


# **10. Find the optimal ARIMA model using Out-of-Time Cross validation** <a class="anchor" id="10"></a>


[Table of Contents](#0.1)


- In Out-of-Time cross-validation, we move backwards in time and forecast into the future to as many steps we took back. Then we compare the forecast against the actuals. 

- To do so, we will create the training and testing dataset by splitting the time series into 2 contiguous parts in a reasonable proportion based on time frequency of series.


- Now, we will build the ARIMA model on training dataset, forecast and plot it.


- From the above chart, the ARIMA(1,1,1) model seems to predict a correct forecast. The actual observed values lie within the 95% confidence band. 

- But, we can see that the predicted forecasts is consistently below the actuals. That means, by adding a small constant to our forecast, the accuracy will certainly improve. 

- So, in this case, we should increase the order of differencing to two (d=2) and iteratively increase p and q up to 5 to see which model gives least AIC and also look for a chart that gives closer actuals and forecasts.

- While doing this, I keep an eye on the P values of the AR and MA terms in the model summary. They should be as close to zero, ideally, less than 0.05.


- The AIC has reduced to 245 from 843 which is good. Mostly, the p-values of the X terms are less than < 0.05, which is great. So overall this model is much better.


# **11. Accuracy Metrics for Time Series Forecast** <a class="anchor" id="11"></a>


[Table of Contents](#0.1)


The commonly used accuracy metrics to judge forecasts are:

1. Mean Absolute Percentage Error (MAPE)
2. Mean Error (ME)
3. Mean Absolute Error (MAE)
4. Mean Percentage Error (MPE)
5. Root Mean Squared Error (RMSE)
6. Lag 1 Autocorrelation of Error (ACF1)
7. Correlation between the Actual and the Forecast (corr)
8. Min-Max Error (minmax)


Typically, we will use three accuracy metrices:-

1. MAPE
2. Correlation and 
3. Min-Max Error 


can be used. The above three are percentage errors that vary between 0 and 1. That way, we can judge how good is the forecast irrespective of the scale of the series.


- Around 23.22% MAPE implies the model is about 76.78% accurate in predicting the next 15 observations. Now we know how to build an ARIMA model manually. 

- But, we should also know how to automate the best model selection process. So, we will discuss it next.


# **12. Auto Arima Forecasting in Python** <a class="anchor" id="12"></a>


[Table of Contents](#0.1)


- In Python, the `pmdarima` package provides `auto_arima()` function which can be used to automate the process of ARIMA Forecasting in Python.

- `auto_arima()` uses a stepwise approach to search multiple combinations of p,d,q parameters and chooses the best model that has the least AIC.

- We need to install the `pmdarima` package first.


# **13. How to interpret the residual plots in ARIMA model** <a class="anchor" id="13"></a>


[Table of Contents](#0.1)

Let’s review the residual plots using stepwise_fit.


## **Interpretation of plots in plot diagnostics**


**Standardized residual**: The residual errors seem to fluctuate around a mean of zero and have a uniform variance.


**Histogram**: The density plot suggest normal distribution with mean slighlty shifted towards right.


**Theoretical Quantiles**: Mostly the dots fall perfectly in line with the red line. Any significant deviations would imply the distribution is skewed.


**Correlogram**: The Correlogram, (or ACF plot) shows the residual errors are not autocorrelated. The ACF plot would imply that there is some pattern in the residual errors which are not explained in the model. So we will need to look for more X’s (predictors) to the model.


Overall, the model seems to be a good fit. So, let's use it to forecast.


# **14. SARIMA model in python** <a class="anchor" id="14"></a>


[Table of Contents](#0.1)


- The plain ARIMA model has a problem. It does not support seasonality.


- If the time series has defined seasonality, then we should go for **Seasonal ARIMA** model (in short **SARIMA**) which uses seasonal differencing.


- Seasonal differencing is similar to regular differencing, but, instead of subtracting consecutive terms, we subtract the value from previous season.


- So, the model will be represented as **SARIMA(p,d,q)x(P,D,Q)**, where, P, D and Q are SAR, order of seasonal differencing and SMA terms respectively and 'x' is the frequency of the time series. If the model has well defined seasonal patterns, then enforce D=1 for a given frequency ‘x’.


- We should set the model parameters such that D never exceeds one. And the total differencing ‘d + D’ never exceeds 2. We should try to keep only either SAR or SMA terms if the model has seasonal components.


- Now, we will build a SARIMA model on the time series dataset.

- But, first import the dataset


- We can see that, the seasonal spikes are intact after applying usual differencing (lag 1). Whereas, it is rectified after seasonal differencing.

- Now, let’s build the SARIMA model using pmdarima‘s `auto_arima()`. To do so, we need to set seasonal=True, set the frequency m=12 for month wise series and enforce D=1.


The model has estimated the AIC and the P values of the coefficients look significant. Let’s look at the residual diagnostics plot.

The best model SARIMAX(3, 0, 0)x(0, 1, 1, 12) has an AIC of 528.6 and the P Values are significant.

Let’s forecast for the next 24 months.


There you have a nice forecast that captures the expected seasonal demand pattern.


# **15. SARIMAX model with exogeneous variables** <a class="anchor" id="15"></a>


[Table of Contents](#0.1)


- Now, we will force an external predictor, also called, `exogenous variable` into the model. This model is called the `SARIMAX model`. The only requirement to use an exogenous variable is we should know the value of the variable during the forecast period as well.

- I want to see how the model looks if we force the recent seasonality pattern into the training and forecast. The seasonal index is a good exogenous variable because it repeats every frequency cycle, 12 months in this case.

- So, we will always know what values the seasonal index will hold for the future forecasts.

- Let’s compute the seasonal index so that it can be forced as a (exogenous) predictor to the SARIMAX model.


The exogenous variable (seasonal index) is ready. Let’s build the SARIMAX model.


# **16. References** <a class="anchor" id="16"></a>


[Table of Contents](#0.1)


The ideas and codes in this notebook are taken from the following websites.


1. https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

2. https://www.geeksforgeeks.org/python-arima-model-for-time-series-forecasting/


So, now we will come to the end of this kernel.

I hope you find this kernel useful and enjoyable.

Your comments and feedback are most welcome.

Thank you


[Go to Top](#0)


### Context:
## **Routine set up**

### Code:
```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

### Context:
## **Import data**

### Code:
```python
path = '/kaggle/input/dataset/dataset.txt'

df = pd.read_csv(path)

df.head()
```

### Context:
- Now, we will continue with our example.

### Code:
```python
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
```

### Context:
- Since p-value(1.00) is greater than the significance level(0.05), let’s difference the series and see how the autocorrelation plot looks like.

### Code:
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})


# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()
```

### Context:
# **6. How to find the order of the AR term (p)** <a class="anchor" id="6"></a>

[Table of Contents](#0.1)


- The next step is to identify if the model needs any AR terms. We will find out the required number of AR terms by inspecting the **Partial Autocorrelation (PACF) plot**.


- **Partial autocorrelation** can be imagined as the correlation between the series and its lag, after excluding the contributions from the intermediate lags. So, PACF sort of conveys the pure correlation between a lag and the series. This way, we will know if that lag is needed in the AR term or not.


- Partial autocorrelation of lag (k) of a series is the coefficient of that lag in the autoregression equation of Y.


$$Yt = \alpha0 + \alpha1 Y{t-1} + \alpha2 Y{t-2} + \alpha3 Y{t-3}$$


- That is, suppose, if Y_t is the current series and Y_t-1 is the lag 1 of Y, then the partial autocorrelation of lag 3 (Y_t-3) is the coefficient $\alpha_3$ of Y_t-3 in the above equation.


- Now, we should find the number of AR terms. Any autocorrelation in a stationarized series can be rectified by adding enough AR terms. So, we initially take the order of AR term to be equal to as many lags that crosses the significance limit in the PACF plot.

### Code:
```python
# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.value.diff().dropna(), ax=axes[1])

plt.show()
```
