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
	df["{}ma".format(str(window))] = df["Adj Close"].rolling(window = window, min_periods = 0).mean()
	#df.dropna(inplace = True)
	ax5 = plt.subplot2grid((6,1), (0,0), rowspan = 4, colspan = 1)
	ax6 = plt.subplot2grid((6,1), (4,0), rowspan = 2, colspan = 1, sharex = ax5)
	ax5.plot(df.index, df["Adj Close"])
	ax5.plot(df.index, df["{}ma".format(str(window))])
	ax6.bar(df.index, df["Volume"])
	#fig3.autofmt_xdate()  #once I add this line, x-axis vanishes
	plt.show()

print(moving_average(7))


#--------------------------------------------------------------------------------------------------------------------------------------
#Mandla's part

# Importing Packages

import pandas as pd
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

# Plot Styles

plt.style.use('fivethirtyeight')

# Opening Statement 

Today = dt.date.today()

print(f"""\n\nThis program is for a daily series as at {Today}, the price of a stock used is the
closing price, which is the last transaction price of the stock in a trading day.\n""")

# The Menu for the client

print("=" * 100, "\n")

print("Please advise your preferred form of loading data? :\n")
print("Option :1 - Loading the data through Copy and Paste. \nOption :2 - Loading the data\
 through a file. \nOption :3 - Loading the Data from a website. \nOption :4 - To quit).", end='\n')
DLoad = int(input('\n       :  > .....'))

print("\n", "=" * 100, "\n")
  
while DLoad != 4:
        
    if DLoad == 1       :
              
        print("\nPlease copy the data in your file now & Key 1 to confirm having copied your file", end='\n')
        DLoad = int(input('\n       :  > .....'))
                
        mydata = pd.read_clipboard(header=0, parse_dates=[0], index_col=0, squeeze=True)
        print("\nThank You, This is the 1st few lines of your data\n")
        print(mydata.head())
        DLoad = int(input('\nInput 4 to affirm completion   :  > .....'))
            
    elif DLoad == 2     :
                
        #FName = ("daily_adjusted_MSFT")
        FName = input("\nPlease state file name : ... ")                      # TODO Remove when going on production
                
        if exists(FName) == False :                                             # To test availability of Data 
            print(f"\nConfirmed, Your File is Available")
        else                        :
            print("\nWe do not have data on this File, Please key in an alternative file", end = "")
            FName = input("\nPlease state file name : ... ")   
                
        mydata = pd.read_csv(FName + ".csv", header=0, parse_dates=['timestamp'], squeeze=True)
        
        print("\nThank You, This is the 1st few lines of your data\n")
        print(mydata.head())
        DLoad = int(input('\nInput 4 to affirm completion   :  > .....'))
            
    elif DLoad == 3     :
                
        Dweb = input("\nPlease state file location : ... ")
        mydata = pd.read_csv(Dweb + ".csv", header=0, parse_dates=[0], index_col=0, squeeze=True)
        print("\nThank You, This is the 1st few lines of your data\n")
        print(mydata.head())
        DLoad = int(input('\nInput 4 to affirm completion   :  > .....'))
    
    elif DLoad == 4     : break                                                 # TODO
    
print("\n", "=" * 100, "\n")

# Data Types 

print("\n", "=" * 100, "\n")

print("\nThe types of data you have per column is : \n")
print(mydata.dtypes)                                                           # To check data types

# Index Slicing 

idx = pd.IndexSlice
#mydat = mydata.loc[idx['2018-01-11':'2018-05-11'], :]
#mydata["timestamp"] = mydata.index.astype("datetime64[ns]")
#print(mydat)
# mydat.loc[idx['2018-01-11']:]                                                 # All Data until 1 Nov 2018 (End Date)

# For capturing the start date and End dates                                    # TODO

#df = 0
#df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3],'day': [4, 5]})
#pd.to_datetime(df[['year', 'month', 'day']])
#pd.to_datetime(df)

# Descriptive Stats of the Data 

print("\nThe descriptive statistics of your data : \n")
mydat = mydata.close[0:len(mydata)]

print(mydat.describe(), end = "")	                                             # To get descriptive stats
print("\n")
print("\n", "=" * 100, "\n")

## Max and Mix Rows 

print("\nThis is your maximum Price from the data : ", mydata.close.max())

Pmax = mydata.close.max()
print("\nThe complete details for the highest price point", mydata.iloc[np.where(mydata.close == Pmax)])

print("\nThis is your minimum Price from the data : ", mydata.close.min())

Pmin = mydata.close.min()
print("The complete details for the lowest price point", mydata.iloc[np.where(mydata.close == Pmin)])
print("\n", "=" * 100, "\n")

## Plotting the Data                  TODO Review

print('\nThe Graph of Your Data Looks Like\n')
plot = mydata.close.plot(figsize = (15,6))
plot.set_xlabel("Days")
plot.set_ylabel("Closing Share Price")
plot.set_title("Share Price Reported per Day")
plot.get_figure().savefig("Closing_Price.pdf")
plt.show()
print("\n", "=" * 100, "\n")

#
#mydata      = mydata.set_index(pd.DatetimeIndex(mydata['timestamp']))
y           = mydata.close[0:len(mydata)]
##my_start    = pd.to_datetime('2018-07-01')
#my_start    = ('2018-07-01')
#my_start
#print(type(my_start))
#
x           =  mydata["timestamp"].astype("datetime64[ns]")

## Linear Regression                    TODO Review

print('\nA Liear Regression Params of Your Data Looks Like\n')
y = mydata.close[:]
x = range(len(mydata))
x = sm.add_constant(x)
results = sm.OLS(y,x).fit()
print('\nThese are the Result Parameters\n')
print(results.params,"\n")
print("\n", "=" * 100, "\n")

# Running an ARIMA Model

print("\nRunning an Autoregressive Model - 'ARIMA'")
model = sm.tsa.statespace.SARIMAX(y, order=(1, 1, 1))    	                # Running an ARIMA (1, 1, 1)
results = model.fit()                                                           # Fitting 
print(results.summary().tables[1]) 				                                # To get the table of coefficients
results.plot_diagnostics(figsize=(15, 12))			                            # Diagnostic plot
plt.show()
print("\n", "=" * 100, "\n")

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


