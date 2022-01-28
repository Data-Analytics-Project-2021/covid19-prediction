from pyexpat import model
import pandas as pd


def fill_date_gaps(df, *, method, dates_range) -> None:
    """
    Fills missing dates within range with method specified 
    and ones before that with 0
    """

    df.loc[dates_range[0]:dates_range[-1]].fillna(method=method, inplace=True)
    df.fillna(0, inplace=True)


def train_test_split(df, *, fraction) -> tuple:
    """
    Split df by fraction into train and test
    """

    percent = int(len(df)*fraction)

    try:
        train = df.iloc[:percent].dropna()
        test = df.iloc[percent:]
    except:
        train = df[:percent]
        test = df[percent:]

    return train, test

def get_all_attrs(obj):
    """
    Print all attributes of an object
    """

    for attr in dir(obj):
        if not attr.startswith('_'):
            print(attr)

def get_varma_model_name(model_fit):
    """
    Return VARMA model name
    """
    model_string = 'V'

    p, q = model_fit.model.order

    if p > 0:
        model_string += 'AR'
    if q > 0:
        model_string += 'MA'

    if p > 0 and q > 0:
        model_string += str(model_fit.model.order)

    elif p > 0:
        model_string += f'(%d)' % p

    elif q > 0:
        model_string += f'(%d)' % q

    else:
        model_string += '(0, 0)'

    return model_string