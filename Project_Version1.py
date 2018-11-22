import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates   #matplotlib does not use datetime
from scipy import stats
import datetime as dt
import time
import fix_yahoo_finance as yf 
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
from mpl_finance import candlestick_ohlc
import copy

yf.pdr_override()

#input data
#df is the dataframe with Datetime index
df = pdr.get_data_yahoo("MSFT", start = "2017-06-01", end = "2018-11-19")
pd.set_option("display.width", None)
#print(df.head())

#df1 is the dataframe with non-datetime index
df1 = copy.deepcopy(df)
df1.reset_index(inplace = True)
print(df1.head())
  
date = df1["Date"]
close = df1["Close"]

#descriptive
def describe_stock(y): 
	mean = y.mean()
	return "Mean: ", mean
	return "Median: ", y.median()
	return "25% quantile: ", y.quantile(.25)
	return "75% quantile: ", y.quantile(.75)
	return "Range: [ ", y.max(),', ', y.min(), " ]"
	std = y.std()
	return "Standard variation: ", std
	return "Coefficient of variation: %.2f%%" % (std/mean*100)   #CV is defined as the ratio of the sd to the mean  

## visualisation

#raw time-series
def time_series(date, close):	
	ax1 = plt.subplot2grid((2,1), (1,0), rowspan = 1)
	ax1.grid()
	plt.plot(date, close)
	plt.xlabel("Date")
	plt.ylabel("Stock Prices")
	plt.title("Raw Time-series")

#trendline  
def trendline(date, close):
	mdate = date.map(mdates.date2num)   #transfer Date to mdates which is a type can be used to plot
	ax2 = plt.subplot2grid((2,1), (0,0), rowspan = 1)
	ax2.grid()
	plot = plt.plot(mdate, close)  
	z = np.polyfit(mdate, close, 1)   #get coefficents
	p = np.poly1d(z)   # get the formular
	plt.plot(mdate, p(mdate), "r")
	#plt.xlabel("Timestamp")
	plt.ylabel("Stock Prices")
	plt.title("Trendline")
	ax2.xaxis_date()   #use it so the graph can show dates, instead of mdates

# stock comparison
def comparison(ticker, startdate, enddate):
	df2 = pdr.get_data_yahoo(ticker, start = startdate, end = enddate)
	df2.reset_index(inplace = True)
	x1 = df2["Date"]
	y1 = df2["Close"]
	fig= plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(x1, y1, color = "b")
	ax1.set_ylabel("Y values for {}".format(ticker), color = "b")
	ax2 = ax1.twinx()
	ax2.plot(x1, close, color = "g")
	ax2.set_ylabel("Y values for MSFT", color = "g")
	ax1.set_title("Stock Price comparison between MSFT and {}".format(ticker))
	fig.autofmt_xdate()
	plt.legend()
	plt.show()

print(comparison("TSLA", "2017-06-01", "2018-11-19"))

# Candlestick
def candlestick(window):    #window can be chosen according to the interest of users
	fig2 = plt.figure()
	#resample can shrink the dataset significantly
	df_ohlc = df["Adj Close"].resample(window).ohlc()    #get the open, high, low, close price date of the windows (e.g. "10D")
	df_volume = df["Volume"].resample(window).sum()   # Note that df_ohlc and df_volume can only be valid only with DatetimeIndex
	df_ohlc.reset_index(inplace = True)  #convert datetimeIndex into a pandas.series
	df_ohlc["Date"]= df_ohlc["Date"].map(mdates.date2num) # convert datetime series into mdate format
	ax3 = plt.subplot2grid((6,1), (0,0), rowspan = 4, colspan = 1)
	ax4 = plt.subplot2grid((6,1), (5,0), rowspan = 2, colspan = 1, sharex = ax3)
	candlestick_ohlc(ax3, df_ohlc.values, width = 2, colorup = 'g')
	ax4.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0) #fill_between(x, y)  x is index datetime (convert into mdate)
	ax4.xaxis_date()  #convert mdates in a_axis into datetime again, for user-friendly
	plt.title("Candlestick")
	fig2.autofmt_xdate()
	plt.show()

# show both time-series and trendline
fig1 = plt.figure()
print(time_series(date,close))
print(trendline(date,close))
fig1.autofmt_xdate()  #to beautify the fig
plt.show()

print(candlestick("10D"))

#Moving average
def moving_average(window):
	fig3 = plt.figure()
	df["{}ma".format(str(window))] = df["Close"].rolling(window = window, min_periods = 0).mean()
	#df.dropna(inplace = True)
	ax5 = plt.subplot2grid((6,1), (0,0), rowspan = 4, colspan = 1)
	ax6 = plt.subplot2grid((6,1), (4,0), rowspan = 2, colspan = 1, sharex = ax5)
	ax5.plot(df.index, df["Adj Close"])
	ax5.plot(df.index, df["{}ma".format(str(window))])
	ax6.bar(df.index, df["Volume"])
	#fig3.autofmt_xdate()  #once I add this line, x-axis vanishes
	plt.show()

print(moving_average(7))

def ExpMovingAverage(values, window):
	fig3 = plt.figure()
	weights = np.exp(np.linspace(-1.,0.,window))
	weights /= weights.sum()
	a= np.convolve(values, weights,mode="full")[:len(close)]
	a[:window]= a[window]
	ax7 = plt.subplot2grid((6,1), (0,0), rowspan = 4, colspan = 1)
	ax8 = plt.subplot2grid((6,1), (4,0), rowspan = 2, colspan = 1, sharex = ax7)
	ax7.plot(df.index, df["Adj Close"])
	ax7.plot(df.index, a)
	ax8.bar(df.index, df["Volume"])
	plt.title("Exponential Moving Average")
	#fig3.autofmt_xdate()
	plt.show()
	
print(ExpMovingAverage(close, 10))

#--------------------------------------------------------------------------------------------------------------------------------------
#Mandla's part

import pandas as pd
import numpy as np
import statsmodels.api as sm
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pylab import rcParams
import warnings
import itertools
import math
import datetime as dt
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import datetime

def setDateIndex(start_date, dataframe):                                        # Resetting Index as a Date
	dataframe.reset_index(inplace = True, drop = True)
	dataframe["Date"] = dataframe.index

	j = dataframe.index[1]
	isSuccess = False
	for i in dataframe.index:

		isSuccess = False
		while not isSuccess:
			add_date = (start_date + datetime.timedelta(j)).weekday()
			if add_date < 5:
				dataframe.loc[i, "Date"] = start_date + datetime.timedelta(j)
				j += 1
				isSuccess = True
			else:
				j += 1

	dataframe.set_index('Date',inplace=True, drop=True)
	return dataframe

def returnStartDateAsNum(dt, x):                                                # Converting Dates to Intergers
	start_date = x[0]
	days = 0
	while(not start_date == dt):
		if start_date.weekday() < 5:
			days += 1
		start_date += datetime.timedelta(1)

	return days

plt.style.use('fivethirtyeight')
yf.pdr_override()

# Importing the Data                                                            # Importation of Data from Yahoo

df = pdr.get_data_yahoo('MMM', start='2018-01-01', end='2018-11-01')
df.reset_index(inplace = True)

x = [i.to_pydatetime() for i in df["Date"]]                                     # Creating x vector (Dates)
y = pd.Series(df["Close"].values, index=x).to_frame()                           # Creating y vector (Closing Prices)

# Running an ARIMA Model - See Word Document for additional Context

print("\nRunning an Autoregressive Model - 'ARIMA'")
model = sm.tsa.statespace.SARIMAX(y, trend='n', order=(2,1,2), seasonal_order=(0,1,1,12))
results = model.fit(disp = 0)                                                   # Fitting the model
print(results.summary().tables[1]) 				                # Parameters
results.plot_diagnostics(figsize=(15, 12))			                # Diagnostic plot - visual confirmation of fit
plt.show()                                                                      # Remove comment TODO

print("\n", "=" * 100, "\n")

user_enter_start_date = datetime.datetime(2018, 4, 1)                           # Forecast Start Date
user_enter_end_date   = datetime.datetime(2019, 2, 1)                           # Target Prediction Date

start_d_num           = returnStartDateAsNum(user_enter_start_date, x)          # See 2 functions above
end_d_num             = returnStartDateAsNum(user_enter_end_date, x)

# Forecasting / Prediction

pred = results.get_prediction(start= start_d_num, end=end_d_num, dynamic=False) # Prediction function using the speced model above
pred_ci = pred.conf_int()                                                       # Prediction confidence Interval

pred_ci   = setDateIndex(x[start_d_num], pred_ci)
pred_mean = setDateIndex(x[start_d_num], pred.predicted_mean.to_frame())

 # Training Data / Prediction Plot

ax = y.plot(label='observed', figsize=(20, 15))
pred_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Share Returns')
plt.title("Plotting Observed against Forecast")
plt.legend()
plt.show()

# Predicted Price                                                               # Predicted Price at end of forecast period

print("Predicted Price is : ", pred_mean.iloc[-1])

#---------------------------------------------------------------------------------------------------------------------------------
#Shweta's part
# 1. random forest with very low accuracy rate 
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from scipy import stats
import datetime as dt
import time
import fix_yahoo_finance as yf 
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

yf.pdr_override()

#input data
df = pdr.get_data_yahoo("MSFT", start = "2018-01-01", end = "2018-6-30")
print(df.head())
data=pd.DataFrame({"High":df.High,"Low":df.Low,"Open":df.Open, "Close":df.Close})    # creating a data frame with cols used for prediction
print(data.head())                                                                   # just to check everything functions(not part of prog)
X=data[["High", "Low", "Open", "Close"]]                                             # this will be used for testing and training part
y=data["Close"]                                                                       # this will be used for testing and training 
y=pd.factorize(data["Close"])[0]                                                      #converetd 2d into 1d array
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3)                  #splitting based on testing and training variable
clf=RandomForestClassifier(n_estimators=10)                                           #creating the random forest classifier 
clf.fit(X_train,y_train)                                                               # valuew of X_train will will go y_train
y_pred=clf.predict(X_test)                                                             #predicting based on test variables
print("Accuracy :", metrics.accuracy_score(y_test,y_pred))                         #calculating the accuracy rate by comparing testing and predicting variables


#2. updated random forest, but end lines are not complete because of error
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from scipy import stats
import datetime as dt
import time
import fix_yahoo_finance as yf 
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

yf.pdr_override()

#input data
df = pdr.get_data_yahoo("MSFT", start = "2018-01-01", end = "2018-6-30")
print(df.head())
data=pd.DataFrame({"High":df.High,"Low":df.Low,"Open":df.Open, "Close":df.Close})
print(data.head())

X=data[["High", "Low", "Open", "Close"]]
y=data["Close"]
print(X.head())
print(y.head())
data["is_train"]=np.random.uniform(0,1,len(data)) <= 0.70
print(data.head())
train,test = data[data["is_train"]==True], data[data["is_train"]==False]
print("training data :", len(train))
print("testing data :", len(test))
y=pd.factorize(train["Close"])[0]
print(y)
clf=RandomForestClassifier(n_jobs=2, random_state=10)
clf.fit((train),y)
pred=clf.predict(test)
print("Accuracy :", metrics.accuracy_score(test,pred))

3. Moving average
def movingaverage():
    z=int(input("enter the window for moving average"))    # taking input for the range of moving average
    x = pd.DataFrame(close.rolling(z).mean())              # calculating moving avg
    print(x)
    plot= x.plot()
    plot.set_xlabel("Date")
    plot.set_ylabel("Moving average of stock")
    plot.set_title("moving averages")
    plt.show()
# 4 exponential avg			
def ema(window):
	weights = np.exp(np.linspace(-1.,0.,window))
	weights /= weights.sum()
	a= np.convolve(close.Close,weights,mode="full")[:len(close)]
	a[:window]= a[window]
	return a


print(ema(5))
#macd
def macd(value, lower=26, upper=12):
	emaupper= ExpMovingAverage(value, upper)
	emalower= ExpMovingAverage(value, lower)
	a= emaupper
	b= emalower
	c= emaupper - emalower
	fig4= plt.figure()
	ax2= plt.subplot2grid((6,4),(5,0), rowspan=1, colspan=4, axisbg="#07000d")
	ax2.spines["bottom"].setcolor("#5998ff")
	ax2.spines["top"].setcolor("#5998ff")
	ax2.spines["left"].setcolor("#5998ff")
	ax2.spines["right"].setcolor("#5998ff")
	ax2.tick_params(axis ="x",colors="w")
	ax2.tick_params(axis ="y",colors="w")
	    

