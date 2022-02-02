import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def plot_dataframe(df, *, title, ylabel='', xlabel='Date', figpath=None, color='k'):
    """
    Plot daily cases/cumulative vaccinations
    """
    # Set the locator
    locator = mdates.MonthLocator(interval=4)  # every 2 months
    # Specify the format - %b gives us Jan, Feb...
    fmt = mdates.DateFormatter('%b-%y')

    plt.figure(figsize=(5,5), dpi=100)


    plt.plot(df, color=color)

    fig = plt.gcf()
    ax = plt.gca()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    plt.grid(True)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Locate months
    ax.xaxis.set_major_locator(locator)

    # Specify formatter
    ax.xaxis.set_major_formatter(fmt)
    
    plt.xlabel=xlabel
    plt.ylabel=ylabel

    plt.title(title)
    # plt.legend(loc='best', fontsize=8)
    plt.show()
    
    
    if figpath is not None:
        fig.savefig(figpath, format='eps', bbox_inches='tight')


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
    plt.grid(linewidth=0.4)
    plt.show()
    
    
    if figpath is not None:
        fig.savefig(figpath, format='eps', bbox_inches='tight')


def plot_fore_test(*, test, fore, title):
    """
    Plot test vs forecasted values
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)

    ax.plot(test, color='blue', label='Test')
    ax.plot(fore, color='red', label='Forecast')
    ax.legend(loc='best')
    plt.title(title)
    plt.show()



def plot_side_by_side(*, train, train_label='Train', test=None, test_label='Test'):
    """
    Plots n-dimensional data in two columns
    """
        
    fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8,4))
    for i, ax in enumerate(axes.flatten()):
        
        if test is not None:
            # Train and test
            ax.plot(train[train.columns[i]], color='blue', label=train_label, linewidth=1)
            ax.plot(test[test.columns[i]], color='red', label=test_label, linewidth=1)

            ax.legend(loc='best')

        else:
            # Only train    
            ax.plot(train[train.columns[i]], color='blue', linewidth=1)
            
        # Decorations
        ax.set_title(train.columns[i])

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
    
    plt.tight_layout()