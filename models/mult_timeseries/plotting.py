import matplotlib.pyplot as plt
import pandas as pd


def plot_train_test_fore(*, train, test, fore, conf=None, title='Forecast vs Actuals', ylabel='', xlabel='Date', figpath=None, start_date=None):
    """
    Plot train, test, forecasted values
    """

    # Confidence of cases
    if conf is not None:
        lower_series = pd.Series(conf[:, 0], index=test.index)
        upper_series = pd.Series(conf[:, 1], index=test.index)
        plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)

    plt.figure(figsize=(5,5), dpi=100)

    # train.loc[start_date:].plot(label='training', color='k')
    # test.plot(label='testing', color='b')
    # fore.plot(label='forecast', color='r')

    plt.plot(train.loc[start_date:], label='training', color='k')
    plt.plot(test, label='actual', color='b')
    plt.plot(fore, label='forecast', color='r')

    fig = plt.gcf()
    
    plt.xlabel=xlabel
    plt.ylabel=ylabel

    plt.title(title)
    plt.legend(loc='upper left', fontsize=8)
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