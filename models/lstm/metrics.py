from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

def MAPE(Y_actual,Y_Predicted, title):
    '''
    Calculate MAPE given the actual and forecast timeseries
    Specify title in the format "{Long/Short} Term {modelname} {countryname}
    Ex Usage: MAPE(val['Confirmed'], ar_fc_series, title="Long-term AR")
    '''
    mask = Y_actual != 0
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual)[mask])*100
    print(f"MAPE of {title} is {mape:.3f}%")

def MAE(Y_actual,Y_Predicted, title, sci_not=False):
    '''
    Calculate MAE given the actual and forecast timeseries
    Specify title in the format "{Long/Short} Term {modelname} {countryname}
    sci_not=True will output the metric in scientific notation format (Ex: 1.233E+11)
    Ex Usage: MAE(val['Confirmed'], ar_fc_series, title="Long-term AR")
    '''
    if sci_not:
        print(f'MAE of {title} is {mean_absolute_error(Y_actual, Y_Predicted):.5e}')
    else:
        print(f'MAE of {title} is {mean_absolute_error(Y_actual, Y_Predicted)}')

def MSE(Y_actual,Y_Predicted, title, sci_not=False):
    '''
    Calculate MSE given the actual and forecast timeseries
    Specify title in the format "{Long/Short} Term {modelname} {countryname}
    sci_not=True will output the metric in scientific notation format (Ex: 1.233E+11)
    Ex Usage: MSE(val['Confirmed'], ar_fc_series, title="Long-term AR")
    '''
    if sci_not:
        print(f'MSE of {title} is {mean_squared_error(Y_actual, Y_Predicted):.5e}')
    else:
        print(f'MSE of {title} is {mean_squared_error(Y_actual, Y_Predicted)}')

def RMSE(Y_actual,Y_Predicted, title, sci_not=False):
    '''
    Calculate RMSE given the actual and forecast timeseries
    Specify title in the format "{Long/Short} Term {modelname} {countryname}
    sci_not=True will output the metric in scientific notation format (Ex: 1.233E+11)
    Ex Usage: MSE(val['Confirmed'], ar_fc_series, title="Long-term AR")
    '''
    if sci_not:
        print(f'MSE of {title} is {np.sqrt(mean_squared_error(Y_actual, Y_Predicted)):.5e}')
    else:
        print(f'MSE of {title} is {np.sqrt(mean_squared_error(Y_actual, Y_Predicted))}')