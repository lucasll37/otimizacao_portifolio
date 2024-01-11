import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import yfinance as yf
from pandas.tseries.offsets import BDay
from pypfopt import expected_returns, risk_models
from typing import Dict, List, Tuple


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [10, 5]
plt.rc('figure', autolayout=True)


def backtest(tickers, period, observation_window) -> None:
    start: str = ( pd.to_datetime(period['end']) + BDay(1)).strftime('%Y-%m-%d')
    end: str = (pd.to_datetime(period['end']) + BDay(1 + observation_window['stepsFoward'])).strftime('%Y-%m-%d')

    adjClosePrice: pd.DataFrame = yf.download(tickers, start, end, progress=False)["Adj Close"]
    adjClosePrice.fillna(method = 'ffill', inplace=True)

    pesos: pd.DataFrame = pd.read_csv('./results/portfolio/pesos.csv', index_col='Ticker')

    real_return: pd.Series = expected_returns.mean_historical_return(adjClosePrice)
    cov_matrix: pd.DataFrame = risk_models.sample_cov(adjClosePrice)

    performance: pd.DataFrame = pd.DataFrame(columns=['Weight', 'Risk', 'Return', 'Weighted Return'])
    performance.index.name = 'Ticker'

    for ticker, row in pesos.iterrows():
        performance.loc[ticker, 'Risk'] = np.sqrt(cov_matrix.loc[ticker, ticker])
        performance.loc[ticker, 'Weight'] = row['Peso']
        performance.loc[ticker, 'Return'] = real_return[ticker]

    performance['Weighted Return'] = performance['Return'] * performance['Weight']

    performance.loc['TOTAL', 'Weight'] = performance['Weight'].sum()
    performance.loc['TOTAL', 'Risk'] = 0.02 #np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
    performance.loc['TOTAL', 'Weighted Return'] = performance['Weighted Return'].sum()
        
    performance.to_csv('./results/portfolio/backtest.csv')

    print(f"\n\nPerformance do Portifólio\n \
          \nInício: {start} \
          \nTérmino: {end} \
          \n\nRendimento: {performance.loc['TOTAL', 'Weighted Return'] * 100:.2f} % \
          \nRisco: {performance.loc['TOTAL', 'Risk'] * 100:.2f} % \
          \n\n")


    
if __name__ == '__main__':

    from variables import tickers, period, observation_window

    backtest(tickers,
             period,
             observation_window
             )