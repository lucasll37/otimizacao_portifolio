import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from typing import List, Dict

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [10, 5]
plt.rc('figure', autolayout=True)

def monteCarlo(tickers, period, observation_window, monte_carlo_simulation, SEED, graphics=True):

    random.seed(SEED)
    np.random.seed(SEED)

    for ticker in tickers:
        data: pd.DataFrame = pd.read_csv(f'./results/data/{ticker}.csv', index_col='Date', parse_dates=True)
        label: str = f"{ticker} period {data.index.min().strftime('%Y-%m-%d')} - {data.index.max().strftime('%Y-%m-%d')}"

        adj_close: pd.DataFrame = pd.read_csv(f'./results/prediction/{ticker}/{label}/Adj Close.csv', index_col='Date', parse_dates=True)
        prediction: pd.DataFrame = pd.read_csv(f'./results/prediction/Expected Return.csv', index_col='Ticker')

        ## Validation
        
        drift: float = prediction.loc[ticker, 'Test Drift']
        volatilidade: float = prediction.loc[ticker, 'Test Volatility']

        forecast: List[float] = []
        precos_simulados: Dict[int, List[float]] = dict()

        for n in range(monte_carlo_simulation):
            
            precos_simulados[n] = [data['Adj Close'][-observation_window["stepsFoward"]]]
            
            for i in range(observation_window["stepsFoward"]):
                
                preco_simulado: float = precos_simulados[n][-1] * (1 + random.gauss(drift, volatilidade))
                precos_simulados[n].append(preco_simulado)
            
            forecast.append(precos_simulados[n][-1])

        if graphics:
            plt.title(f'Simulação Monte Carlo Para os Preço em {observation_window["stepsFoward"]} dias (Validação)')
            plt.hist(forecast, density = True, bins=20)
            plt.xlabel('Preço')
            plt.ylabel('Frequência')
            
            if not os.path.exists(f'./results/graphics/{ticker}/{label}/Simulation Monte Carlo'):
                os.makedirs(f'./results/graphics/{ticker}/{label}/Simulation Monte Carlo')
            
            plt.savefig(f'./results/graphics/{ticker}/{label}/Simulation Monte Carlo/[VALIDATION] Return.png')
            plt.close()

            plt.title(f'Simulação Monte Carlo Para os Preços em {observation_window["stepsFoward"]} dias (Validação)')

            soma: List[float] = [0 for _ in range(observation_window["stepsFoward"])]

            for _, value in precos_simulados.items():
                valueSeries: pd.Series = pd.Series(
                    value[:observation_window["stepsFoward"]],
                    index=adj_close[:observation_window["stepsFoward"]].index
                )   
                
                plt.plot(valueSeries, linestyle='--', linewidth=1)
                
                soma: List[float] = [x + y for x, y in zip(soma, value)]

            forecast_mean: List[float] = [x / monte_carlo_simulation for x in soma]
            pd_forecast_mean: pd.Series = pd.Series(forecast_mean, index=adj_close.index[: observation_window["stepsFoward"]])  

            plt.plot(pd_forecast_mean, label = "Média das Previsões", color = 'black', linestyle='-.', linewidth=3)
            plt.plot(data['Adj Close'][-2 * observation_window["stepsFoward"]: - observation_window["stepsFoward"]], color = 'black', label = 'StepsBack', linestyle='-', linewidth=3)
            plt.plot(data['Adj Close'][-observation_window["stepsFoward"]:], color = 'red', label = 'Fact', linestyle='-', linewidth=3)
            plt.plot(adj_close.index[-observation_window["stepsFoward"]], forecast_mean[-1], 'ro')
            plt.text(adj_close.index[-observation_window["stepsFoward"]], forecast_mean[-1], f'{forecast_mean[-1]:.2f}', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
            plt.xticks(rotation=45)
            plt.legend(loc = 'best') 
            plt.savefig(f'./results/graphics/{ticker}/{label}/Simulation Monte Carlo/[VALIDATION] Close Price.png')

            plt.close()

            ## Forecast

            drift: float = prediction.loc[ticker, 'Test Drift']
            volatilidade: float = prediction.loc[ticker, 'Test Volatility']

            forecast: List[float] = []
            precos_simulados: Dict[int, List[float]] = dict()

            for n in range(monte_carlo_simulation):
                
                precos_simulados[n] = [data['Adj Close'][-1]]
                
                for i in range(observation_window["stepsFoward"]):
                    
                    preco_simulado: float = precos_simulados[n][-1] * (1 + random.gauss(drift, volatilidade))
                    precos_simulados[n].append(preco_simulado)
                
                forecast.append(precos_simulados[n][-1])

            plt.hist(forecast, density = True)
            plt.title(f'Simulação Monte Carlo Para os Preço em {observation_window["stepsFoward"]} dias')
            plt.xlabel('Preço')
            plt.ylabel('Frequência')            
            plt.savefig(f'./results/graphics/{ticker}/{label}/Simulation Monte Carlo/Return.png')
            plt.close()


            plt.title(f'Simulação Monte Carlo Para os Preços em {observation_window["stepsFoward"]} dias')

            soma: List[float] = [0 for _ in range(observation_window["stepsFoward"])]

            for _, value in precos_simulados.items():
                valueSeries: pd.Series = pd.Series(
                    value[- observation_window["stepsFoward"]:],
                    index=adj_close.index[- observation_window["stepsFoward"]:])    
                plt.plot(valueSeries, linestyle='--', linewidth=1)
                
                soma: List[float] = [x + y for x, y in zip(soma, value)]

            forecast_mean: List[float] = [x / monte_carlo_simulation for x in soma]
            pd_forecast_mean: pd.Series = pd.Series(forecast_mean, index=adj_close.index[- observation_window["stepsFoward"]:])    

            plt.plot(pd_forecast_mean, label = "Média de Previsões", color = 'black', linestyle='-.', linewidth=3)
            plt.plot(data['Adj Close'][-60:], color = 'black', label = 'StepsBack', linestyle='-', linewidth=3)
            plt.legend(loc = 'best')
            plt.plot(adj_close.index[-1], forecast_mean[-1], 'ro')
            plt.text(adj_close.index[-1], forecast_mean[-1], f'{forecast_mean[-1]:.2f}', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
            plt.xticks(rotation=45)
            plt.savefig(f'./results/graphics/{ticker}/{label}/Simulation Monte Carlo/Close Price.png')
            plt.close()


if __name__ == '__main__':

    from variables import tickers, period, observation_window, monte_carlo_simulation, SEED

    monteCarlo(
        tickers,
        period,
        observation_window,
        monte_carlo_simulation,
        SEED,
        graphics=True
    )
    