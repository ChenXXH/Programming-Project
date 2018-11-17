import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from scipy import stats
import datetime as dt
import time
import fix_yahoo_finance as yf 
import numpy as np

yf.pdr_override()

#input data
df = pdr.get_data_yahoo("MSFT", start = "2018-01-01", end = "2018-6-30")
print(df.head())
close = pd.DataFrame(df.Close)


# did not print out the type of Date

df["ID"] = [i for i in range(len(df.Close))]
df.reset_index(inplace = True)
df.set_index("ID", inplace = True)

data = df[["Date", "Close"]]
print(data.head())

#datetime transfer
	
ts_list = []
for date in data["Date"]:
	timestamp = time.mktime(date.timetuple())
	ts_list.append(timestamp)

data["Timestamp"] = ts_list

print(data.head())

x = data["Timestamp"]
y = data["Close"]
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

close = close.plot()
close.set_xlabel("Date")
close.set_ylabel("Stock Prices")
close.set_title("Raw Time-series")
plt.show()
print(type(close))
#trendline    #how to solve it

#date_list = [date for date in data['Date']]

plot = plt.plot(x, y)
print(type(plot))     #type: list(), then how to change x/y axis??
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
#plot.set_xticklabels(date_list)   I want to change x-axis, but this method won't work on my computer
plt.plot(x, p(x), "r")
plt.show()

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

