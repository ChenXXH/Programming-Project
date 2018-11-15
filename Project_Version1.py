import pandas as pd
import numpy as np

from pandas_datareader import data as pdr 
import fix_yahoo_finance as yf 

yf.pdr_override()

data =  pdr.get_data_yahoo("GOOG", start = "2018-01-01", end = "2018-06-30")

df["ID"] = " "
df.reset_index(inplace = True)
df.set_index("ID", inplace = True)

data = df[["Date", "Close"]]
x = data["Date"]
y = data["Close"]
print(type(data.Date))


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

close.Close.plot()
plt.show()
