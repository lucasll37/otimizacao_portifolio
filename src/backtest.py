import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import yfinance as yf
import matplotlib.dates as mdates
from pandas.tseries.offsets import BDay
from pypfopt import expected_returns, risk_models


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [10, 5]
plt.rc('figure', autolayout=True)


def backtest(tickers, period, observation_window) -> None:
    start: str = ( pd.to_datetime(period['end']) + BDay(1)).strftime('%Y-%m-%d')
    end: str = (pd.to_datetime(period['end']) + BDay(1 + observation_window['stepsFoward'])).strftime('%Y-%m-%d')

    adjClosePrice: pd.DataFrame = yf.download(tickers, start, end, progress=False)["Adj Close"]
    adjClosePrice.fillna(method = 'ffill', inplace=True)

    weights: pd.DataFrame = pd.read_csv('./results/portfolio/weight.csv', index_col='Ticker')

    real_return: pd.Series = expected_returns.mean_historical_return(adjClosePrice, frequency=observation_window['stepsFoward'])
    cov_matrix: pd.DataFrame = risk_models.sample_cov(adjClosePrice)

    performance: pd.DataFrame = pd.DataFrame(columns=['Weight', 'Risk', 'Return', 'Weighted Return'])
    performance.index.name = 'Ticker'

    for ticker, row in weights.iterrows():
        performance.loc[ticker, 'Risk'] = np.sqrt(cov_matrix.loc[ticker, ticker])
        performance.loc[ticker, 'Weight'] = row['Peso']
        performance.loc[ticker, 'Return'] = real_return[ticker]

    performance['Weighted Return'] = performance['Return'] * performance['Weight']

    performance.loc['TOTAL', 'Weight'] = performance['Weight'].sum()
    performance.loc['TOTAL', 'Weighted Return'] = performance['Weighted Return'].sum()
        
    if not os.path.exists(f'./results/backtest'):
        os.makedirs(f'./results/backtest')

    performance.to_csv('./results/backtest/backtest.csv')

    print(f"\n\nPerformance do Portifólio\n \
          \nInício: {start} \
          \nTérmino: {end} \
          \n\nRendimento: {performance.loc['TOTAL', 'Weighted Return'] * 100:.2f} % \
          \n\n")
    
    fig, axes = plt.subplots(nrows=len(weights), ncols=1, figsize=(10, 5 * len(weights)))  # Ajuste o tamanho conforme necessário
    fig.suptitle(f"Backtest - {start} - {end}")

    for i, (ticker, row) in enumerate(weights.iterrows()):
        ax = axes[i]
        ax.plot(adjClosePrice[ticker], label = f'{ticker} - {row["Peso"] * 100:.1f} %') 

        ax.set_ylabel('Valor (R$)')
        ax.set_xlabel('Data')
        ax.legend(loc='upper left')
        ax.set_xticklabels(adjClosePrice.index, rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


    fig.subplots_adjust(top=0.95)  # Ajuste conforme necessário para o título principal

    # Salvando a figura
    plt.savefig(f'./results/backtest/result.png')
    plt.close(fig)

    
if __name__ == '__main__':

    from variables import tickers, period, observation_window

    backtest(tickers,
             period,
             observation_window
             )