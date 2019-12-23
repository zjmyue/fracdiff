import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader
import seaborn

from fracdiff import FracDiff

seaborn.set()
seaborn.set_style('ticks')


def fetch_spx():
    """
    Returns S&P 500 historical price as pandas.Series.
    """
    return pandas_datareader.data.DataReader(
        '^GSPC', 'yahoo', '1998-01-01', '2018-12-31'
    )['Adj Close']


def plot(series_spx, series_spxd):
    """
    Save figures of S&P 500 and its fractional differentiation.
    """
    title = 'S&P 500 and its fractional differentiation'
    fig, ax_spx = plt.subplots(figsize=(16, 8))
    ax_spxd = ax_spx.twinx()

    plot_spx = ax_spx.plot(
        series_spx, color='blue', linewidth=0.6,
        label='S&P 500 (left)'
    )
    plot_spxd = ax_spxd.plot(
        series_spxd, color='orange', linewidth=0.6,
        label='S&P 500, 0.5th differentiation (right)'
    )
    plots = plot_spx + plot_spxd
    labels = [plot.get_label() for plot in plots]

    plt.title(title)
    ax_spx.legend(plots, labels, loc=0)
    plt.savefig('./spx.png', bbox_inches="tight", pad_inches=0.1)


def main():
    series_spx = fetch_spx()

    spxd = FracDiff(0.5, window=1000).transform(series_spx.values)
    series_spxd = pd.Series(spxd, index=series_spx.index)

    plot(series_spx, series_spxd)


if __name__ == '__main__':
    main()
