import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader
import seaborn

from fracdiff import StationaryFracDiff

seaborn.set()
seaborn.set_style('ticks')


def fetch_nky():
    """
    Returns Nikkei 225 historical price as pandas.Series.
    """
    return pandas_datareader.data.DataReader(
        '^N225', 'yahoo', '1998-01-01', '2018-12-31'
    )['Adj Close']


def plot(series_nky, series_nkyd, order):
    """
    Save figures of Nikkei 225 and its fractional differentiation.
    """
    title = 'Nikkei 225 and its fractional differentiation'
    fig, ax_nky = plt.subplots(figsize=(16, 8))
    ax_nkyd = ax_nky.twinx()

    plot_nky = ax_nky.plot(
        series_nky, color='blue', linewidth=0.6,
        label='Nikkei 225 (left)'
    )
    plot_nkyd = ax_nkyd.plot(
        series_nkyd, color='orange', linewidth=0.6,
        label=f'Nikkei 225, {order:.2f}th differentiated (right)'
    )
    plots = plot_nky + plot_nkyd
    labels = [plot.get_label() for plot in plots]

    plt.title(title)
    ax_nky.legend(plots, labels, loc=0)
    plt.savefig('./nky.png', bbox_inches="tight", pad_inches=0.1)


def main():
    series_nky = fetch_nky()

    statfracdiff = StationaryFracDiff(window=1000)
    nkyd = statfracdiff.fit_transform(series_nky.values)
    series_nkyd = pd.Series(nkyd, index=series_nky.index)
    order = statfracdiff.order_

    plot(series_nky, series_nkyd, order)


if __name__ == '__main__':
    main()
