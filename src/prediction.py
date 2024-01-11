import warnings
warnings.filterwarnings('ignore')

from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import random
from joblib import load
from pandas.tseries.offsets import BDay
from pypfopt import expected_returns
from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [10, 5]
plt.rc('figure', autolayout=True)


def prediction(
    tickers: List[str], 
    period: dict, 
    SEED: int, 
    observation_window: dict, 
    graphics: bool = False
) -> None:

    random.seed(SEED)
    np.random.seed(SEED)

    for ticker in tickers:
        data: pd.DataFrame = pd.read_csv(f'./results/data/{ticker}.csv', index_col='Date', parse_dates=True)
        label: str = f"{ticker} - trained: {data.index.min().strftime('%Y-%m-%d')} -> {data.index.max().strftime('%Y-%m-%d')}"

        df_test: pd.DataFrame = data[data.index >= pd.Timestamp(period['boundary'])][['Adj Close']]

        scaler: MinMaxScaler = load(f'./results/serialized objects/{ticker}/scaler - {label}.joblib')
        model: keras.models.Model = keras.models.load_model(f'./results/trained models/{ticker}/{label}.h5')

        X_test: pd.DataFrame = data[['Adj Close']].iloc[-observation_window['stepsBack'] - observation_window['stepsFoward'] : - observation_window['stepsFoward']]
        scaler_X_test: np.ndarray = scaler.transform(X_test)
        scaled_adj_close_test: np.ndarray = model.predict(scaler_X_test.reshape(1, -1, 1))
        adj_close_test: np.ndarray = scaler.inverse_transform(scaled_adj_close_test)

        adj_close_test: pd.DataFrame = pd.DataFrame(
            adj_close_test.reshape(-1, 1),
            index=data[-observation_window['stepsFoward']:].index,
            columns=['Adj Close Test']
        )

        X_temp: pd.DataFrame = data[['Adj Close']].iloc[-observation_window['stepsBack']:]
        scaler_X_temp: np.ndarray = scaler.transform(X_temp)
        scaled_adj_close_forecast: np.ndarray = model.predict(scaler_X_temp.reshape(1, -1, 1))
        adj_close_forecast: np.ndarray = scaler.inverse_transform(scaled_adj_close_forecast)

        adj_close_forecast = pd.DataFrame(
            adj_close_forecast.reshape(-1, 1),
            index=pd.bdate_range(start=X_temp.index[-1] + BDay(1), periods=observation_window['stepsFoward']),
            columns=['Adj Close Forecast']
        )

        real_adj_close_test: pd.DataFrame = data[['Adj Close']][-observation_window['stepsFoward']:]
        real_adj_close_test.rename(columns={'Adj Close': 'Real Adj Close Test'}, inplace= True)
        adj_close: pd.DataFrame = pd.concat([real_adj_close_test, adj_close_test, adj_close_forecast])
        adj_close.index.name = 'Date'
        adj_close: pd.DataFrame = adj_close.groupby(adj_close.index).sum()

        if not os.path.exists(f'./results/prediction/{ticker}/{label}'):
            os.makedirs(f'./results/prediction/{ticker}/{label}')

        adj_close.to_csv(f'./results/prediction/{ticker}/{label}/Adj Close.csv')

        try:
            info: pd.DataFrame = pd.read_csv('./results/prediction/Expected Return.csv', index_col='Ticker')

        except:
            info: pd.DataFrame = pd.DataFrame(columns=['Expected Returns', 'Volatility', 'Drift', 'Test Volatility', 'Test Drift'])
            info.index.name = 'Ticker'

        info.loc[ticker, 'Expected Returns'] = expected_returns.mean_historical_return(adj_close_forecast)['Adj Close Forecast']
        info.loc[ticker, 'Volatility'] = X_test[['Adj Close']].pct_change().dropna().std()['Adj Close']
        info.loc[ticker, 'Drift'] = (1 + expected_returns.mean_historical_return(adj_close_forecast)['Adj Close Forecast']) ** (1/len(adj_close_forecast)) - 1
        info.loc[ticker, 'Test Volatility'] = X_test[['Adj Close']].pct_change().dropna().std()['Adj Close']
        info.loc[ticker, 'Test Drift'] = (1 + expected_returns.mean_historical_return(adj_close_test)['Adj Close Test']) ** (1/len(adj_close_forecast)) - 1
        info.to_csv('./results/prediction/Expected Return.csv')

        if graphics:
            plt.title("Previsão do preço de Fechamento Ajustado")
            plt.plot(df_test['Adj Close'][-observation_window['stepsBack']:- observation_window['stepsFoward']], label = 'StepsBack') 
            plt.plot(adj_close['Adj Close Forecast'][-observation_window['stepsFoward']:], label = 'Forecast') 
            plt.plot(adj_close['Adj Close Test'][-2 * observation_window['stepsFoward']:- observation_window['stepsFoward']], label = 'Forecast Test') 
            plt.plot(adj_close['Real Adj Close Test'][-2 * observation_window['stepsFoward']:- observation_window['stepsFoward']], label = 'Expected Forecast Test')
            plt.legend(loc = 'best') 
            plt.xticks(rotation=45)
            plt.savefig(f'./results/prediction/{ticker}/{label}/return.png')
            plt.close()

if __name__ == '__main__':

    from variables import tickers, period, observation_window, SEED

    prediction(tickers, period, SEED, observation_window, graphics=True)