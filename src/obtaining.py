import warnings
warnings.filterwarnings('ignore')

from typing import TypedDict
from datetime import datetime
import os
import yfinance as yf
import numpy as np

class Period(TypedDict):
    start: datetime
    boundary: datetime
    end: datetime

def getData(ticker: str, period: Period) -> None:
    try:
        data = yf.download(ticker, period['start'], period['end'])

        for index, _ in data.iterrows():
            data.loc[index, 'year'] = index.year
            data.loc[index, 'month'] = index.month
            data.loc[index, 'day'] = index.day
            data.loc[index, 'weekday'] = index.dayofweek

        data['Return'] = data['Adj Close'].pct_change()
        data.dropna(inplace=True)
        data['Return'] = data['Return'] + 1
        data['Log Return'] = np.log(data['Return'])
           
        if not os.path.exists(f'./results/data'):
            os.makedirs(f'./results/data')

        data.to_csv(f'./results/data/{ticker}.csv')

    except Exception as e:
        print(f"Erro ao baixar dados: {e}")

if __name__ == '__main__':
    
    from variables import tickers, period

    for ticker in tickers:
        getData(ticker, period)

