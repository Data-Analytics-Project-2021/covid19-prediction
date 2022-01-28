import pandas as pd

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

def cointegration_test(df, alpha=0.05): 
    """
    Perform Johanson's Cointegration Test and Report Summary
    """

    out = coint_johansen(df,-1,5)

    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]

    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)


def run_dicky_fuller(ts):
    """
    Function to run Augmented Dicky Fuller test on the passed 
    time series and report the statistics from the test
    """

    print("Observations of Dickey-fuller test")
    dftest = adfuller(ts,autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])

    for key,value in dftest[4].items():
        dfoutput['critical value (%s)'%key]= value
    print(dfoutput)

