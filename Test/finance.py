
import pandas as pd
import numpy as np
from functools import reduce

df_INTC = pd.read_csv("INTC.csv")
df_CSCO = pd.read_csv("CSCO.csv")
df_QCOM = pd.read_csv("QCOM.csv")
df_EBAY = pd.read_csv("EBAY.csv")
df_AAPL = pd.read_csv("AAPL.csv")
df_AMZN = pd.read_csv("AMZN.csv")

dfs = [df_INTC[['Date', 'Close']], df_CSCO, df_QCOM, df_EBAY,df_AAPL,df_AMZN]
df = reduce(lambda left,right: pd.merge(left,right[['Date', 'Close']],on='Date'), dfs)
df.columns = ['Date','Intel','Cisco','QUALCOMM','eBay','Apple','Amazon']

X_finance = df.ix[:,1:7]
X_finance = X_finance.apply(np.log)          # log
X_finance = X_finance.diff().drop(0)         # stock return