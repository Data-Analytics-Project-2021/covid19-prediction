import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

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