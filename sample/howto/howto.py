import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import pandas_datareader
import seaborn

from fracdiff import Fracdiff, StationaryFracdiff

seaborn.set_style('ticks')
register_matplotlib_converters()


def fetch_price(ticker):
    """Return historical price"""
    return pandas_datareader.data.DataReader(
        ticker, 'yahoo', '1998-01-01', '2018-12-31'
    )['Adj Close']


def plot_spx(spx, spxd):
    plt.title('S&P 500 and its fractional differentiation')
    fig, ax_spx = plt.subplots(figsize=(16, 8))
    ax_spxd = ax_spx.twinx()

    plot_spx = ax_spx.plot(
        spx, color='blue', linewidth=0.6,
        label='S&P 500 (left)'
    )
    plot_spxd = ax_spxd.plot(
        spxd, color='orange', linewidth=0.6,
        label='S&P 500, 0.5th differentiation (right)'
    )
    plots = plot_spx + plot_spxd
    labels = [plot.get_label() for plot in plots]

    ax_spx.legend(plots, labels, loc=0)
    plt.savefig('spx.png', bbox_inches="tight", pad_inches=0.1)


def plot_nky(nky, nkyd, order):
    plt.title('Nikkei 225 and its fractional differentiation')
    fig, ax_nky = plt.subplots(figsize=(16, 8))
    ax_nkyd = ax_nky.twinx()

    plot_nky = ax_nky.plot(
        nky, color='blue', linewidth=0.6,
        label='Nikkei 225 (left)'
    )
    plot_nkyd = ax_nkyd.plot(
        nkyd, color='orange', linewidth=0.6,
        label=f'Nikkei 225, {order:.2f}th differentiated (right)'
    )
    plots = plot_nky + plot_nkyd
    labels = [plot.get_label() for plot in plots]

    ax_nky.legend(plots, labels, loc=0)
    plt.savefig('nky.png', bbox_inches="tight", pad_inches=0.1)


def howto_spx():
    spx = fetch_price('^GSPC')

    window = 100

    fracdiff = Fracdiff(0.5, window=window)
    spx_diff = fracdiff.transform(spx.values.reshape(-1, 1))
    spxd = pd.Series(spx_diff[:, 0], index=spx.index)

    plot_spx(spx[window:], spxd[window:])


def howto_nky():
    nky = fetch_price('^N225')

    window = 100

    statfracdiff = StationaryFracdiff(window=window)
    nky_diff = statfracdiff.fit_transform(nky.values.reshape(-1, 1))
    nkyd = pd.Series(nky_diff[:, 0], index=nky.index)

    plot_nky(nky[window:], nkyd[window:], statfracdiff.order_[0])


def main():
    howto_spx()
    howto_nky()


if __name__ == '__main__':
    main()
