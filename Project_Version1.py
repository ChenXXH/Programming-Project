import pandas as pd
import numpy as np

from pandas_datareader import data as pdr 
import fix_yahoo_finance as yf 

yf.pdr_override()

data =  pdr.get_data_yahoo("GOOG", start = "2018-01-01", end = "2018-06-30")

print(data.Open.describe())