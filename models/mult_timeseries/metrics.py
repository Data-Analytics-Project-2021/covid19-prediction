import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error


# def MAPE(Y_actual,Y_Predicted, title):
#     '''
#     Calculate MAPE given the actual and forecast timeseries
#     Specify title in the format "{Long/Short} Term {modelname} {countryname}
#     Ex Usage: MAPE(val['Confirmed'], ar_fc_series, title="Long-term AR")
#     '''
#     mask = Y_actual != 0
#     mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual)[mask])*100
#     print(f"MAPE of {title} is {mape:.3f}%")

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
        print(f'RMSE of {title} is {np.sqrt(mean_squared_error(Y_actual, Y_Predicted)):.5e}')
    else:
        print(f'RMSE of {title} is {np.sqrt(mean_squared_error(Y_actual, Y_Predicted))}')

def MAPE(Y_actual, Y_Predicted, title=None):
    """
    Calculates mean absolute percentage error
    """
    Y_actual = np.array(Y_actual).flatten()
    Y_Predicted = np.array(Y_Predicted).flatten()
    
    mask = Y_actual != 0
    
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual)[mask])*100
    if title:
        print(f"MAPE of {title} is {mape}%")
    return mape



# evaluate an VARMA model for a given order (p,d,q)
def evaluate_varma_model(train, test, varma_order, column):
    """
    Evaluates a single VARMA model of order varma_order
    """
    
    model = VARMAX(train, order=varma_order, enforce_stationarity=True)
    model_fit = model.fit()

    # Forecast long-term
    yhat = model_fit.forecast(len(test))

    # Evaluate metrics
    mse = mean_squared_error(test[column], yhat[column])
    mape = MAPE(test[column], yhat[column])
    mae = mean_absolute_error(test[column], yhat[column])

    # Return metrics
    return {'mse': mse, 'mape': mape, 'mae': mae}


# evaluate combinations of p and q values for an VARMA model
def evaluate_varma_models(train, test, p_values, q_values, column):
    """
    Grid searches and evaluates all VARMA models with given
    p_values and q_values
    """
    best_score, best_cfg = {'mse': float("inf")}, None
    
    for p in p_values:
        for q in q_values:
            order = (p,q)
            try:
                eval = evaluate_varma_model(train, test, order, column)
                
                if eval['mse'] < best_score['mse']:
                    best_score['mse'], best_cfg = eval['mse'], order
                
                print('VARMA%s: MSE=%.3f, MAPE=%.3f, MAE=%.3f' % (order, eval['mse'], eval['mape'], eval['mae']))
            except:
                continue
    print()
    print('Best VARMA%s: MSE=%.3f' % (best_cfg, best_score['mse']))