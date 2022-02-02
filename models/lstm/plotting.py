import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_train_test_fore(*, train, test, fore, conf=None, title='Forecast vs Actuals', ylabel='', xlabel='Date', figpath=None, start_date='2021-03-01'):
    """
    Plot train, test, forecasted values
    """

    # Set the locator
    locator = mdates.MonthLocator(interval=2)  # every 2 months
    # Specify the format - %b gives us Jan, Feb...
    fmt = mdates.DateFormatter('%b-%y')

    # Confidence of cases
    if conf is not None:
        lower_series = pd.Series(conf[:, 0], index=test.index)
        upper_series = pd.Series(conf[:, 1], index=test.index)
        plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)

    plt.figure(figsize=(5,5), dpi=100)


    plt.plot(train.loc[start_date:], label='training', color='k')
    plt.plot(test, label='actual', color='b')
    plt.plot(fore, label='forecast', color='r')

    fig = plt.gcf()
    ax = plt.gca()
    x = plt.gca().xaxis
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))


    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Locate months
    x.set_major_locator(locator)

    # Specify formatter
    x.set_major_formatter(fmt)
    
    plt.xlabel=xlabel
    plt.ylabel=ylabel

    plt.title(title)
    plt.legend(loc='best', fontsize=8)
    plt.grid(linewidth=0.2, color='gray')
    plt.show()
    
    
    if figpath is not None:
        fig.savefig(figpath, format='eps', bbox_inches='tight')