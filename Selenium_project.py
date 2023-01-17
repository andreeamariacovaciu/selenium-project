from selenium import webdriver
from selenium.webdriver.chrome.service import Service
#from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.arima.model import ARIMAResults
from numpy import log
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
import numpy as np 
from datetime import datetime
import math


def listToString(s): 
    str1 = ""   
    for ele in s: 
        str1 += ele  
    return str1 

def plotdata(dfPrice):
    f=plt.figure()
    ax1=f.add_subplot(121)
    ax1.set_title('Data')
    ax1.plot(dfPrice)

    # ax2=f.add_subplot(122)
    # ax2.set_title('Train and test set')
    # ax2.plot(dfPrice[30:], color = "black")
    # ax2.plot(dfPrice[:30], color = "red")
    # print('test',dfPrice[:30])
    plt.show()
    


def determine_d(dfPrice):
    d=0
    result = adfuller(dfPrice.dropna())
    print('ADF Statistic: %f' % result[0])
    pvalue_org=result[1]
    print('p-value: %f' % result[1])
    if pvalue_org>0.05:
        result = adfuller(dfPrice.diff().dropna())
        pvalue1=result[1]
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        if pvalue1<0.05:
            print('d=1')
            d=1
    return d  

def plot_determine_d(dfPrice):
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(dfPrice, 'r')
    axes[0, 0].set_title('Original Series')
    plot_acf(dfPrice, lags=50, ax=axes[0, 1])
    # 1st Differencing
    axes[1, 0].plot(dfPrice.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(dfPrice.diff().dropna(), lags=60, ax=axes[1, 1])
    plt.show()

def determine_p(dfPrice):
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(dfPrice.diff()); axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0,5)); axes[1].set_title('Partial Autocorrelation Determine p')
    plot_pacf(dfPrice.diff().dropna(), lags=60, ax=axes[1])
    plt.show() 

def determine_q(dfPrice):
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(dfPrice.diff()); axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0,1.2)); axes[1].set_title('Determine q')
    plot_acf(dfPrice.diff().dropna(), lags=60, ax=axes[1])
    plt.show()

##################
##### order=(p,d,q)
def forecast_test(df):
    model = ARIMA(df.Price[:111], order=(1, 1, 1))  
    results=model.fit()
    n_periods=15
    future_dates=[df.index[-1]+x for x in range(0,n_periods)]
    future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
    future_df=pd.concat([df,future_datest_df])
    future_df['Forecast'] = results.predict(start=111, end=125)
    #print('.....',future_df)
    mse=0
    print(future_df['Forecast'])
    for i in range(111,125):
        mse=mse+ (future_df['Forecast'][i]-future_df['Price'][i])**2
    print('Mean Squared Error:', math.sqrt(mse)/n_periods)

    future_df[['Price', 'Forecast']].plot(figsize=(12, 8))
    plt.title('Test Forecasting')
    plt.show()

def forecast(df):
    model = ARIMA(df.Price[:120], order=(1, 1, 2))  
    results=model.fit()
    future_dates=[df.index[-1]+x for x in range(0,10)]
    future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
    future_df=pd.concat([df,future_datest_df])
    future_df['Forecast'] = results.predict(start=126, end=129)
    print('Forecasting:')
    for i in range(126,129):
        print(future_df['Date'][i],future_df['Forecast'][i])
    
    future_df[['Price', 'Forecast']].plot(figsize=(12, 8))
    plt.title('3 days forecasting')
    plt.show()

def main():
    from webdriver_manager.chrome import ChromeDriverManager
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get('https://www.bvb.ro/FinancialInstruments/Details/FinancialInstrumentsDetails.aspx?s=m')
    print(driver.title)
    print('------------')
    trading = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, "//input[@value='Trading']" ))
    )                                                           
    trading.click()

    data_page = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "gvSummar"))
    )                                                           

    data=[]
    data.append(data_page.text)
    i=0
    while i<12:
        next_tab = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, 'gvSummar_next')) 
        ) 
            #print(next_tab.text) 
        driver.execute_script("arguments[0].click();", next_tab)                                                           
        driver.implicitly_wait(10)
        data_page1 = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "gvSummar"))
        ) 
        data.append(data_page1.text)
        i=i+1                                                      

    #List where every element is a row:
    data_list=listToString(data).splitlines()

    #List of list:
    new_data_list=[]
    for elem in data_list:
        new_data_list.append(elem.split(' '))

    date=[]
    price=[]
    for i in range(1,len(new_data_list)):
        date.append(new_data_list[i][0])
        price.append(new_data_list[i][9])

    df=pd.DataFrame()
    df['Date']=pd.to_datetime(date,format='%m/%d/%Y')
    df['Price']=pd.DataFrame(price).astype(float)
    df=df.drop_duplicates(subset=['Date'])
    df.set_index('Date',inplace=True)
    print(df)

    driver.quit()

    ##train=df.Price[:30]
    ##test=df.Price[30:]
    #plotdata(df.Price)
    plot_determine_d(df.Price)
    determine_d(df.Price)
    determine_p(df.Price)
    determine_q(df.Price)

    df_inv= df.reindex(index=df.index[::-1])
    df_inv=df_inv.reset_index()
   # df_inv_train=df_inv[:126]
    print(df_inv)
    #forecast_test(df_inv)
    forecast(df_inv)

if __name__ == '__main__':
  main()