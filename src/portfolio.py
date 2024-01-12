import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models, plotting
from typing import Dict, List, Tuple


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [10, 5]
plt.rc('figure', autolayout=True)


def make_portfolio(tickers: List[str], 
                   period: Dict[str, str], 
                   SEED: int, 
                   risk_free_rate: float, 
                   minimum_participation: float,
                   maximum_participation: float, 
                   use_ia: bool) -> None:

    random.seed(SEED)
    np.random.seed(SEED)

    adjClosePrice: pd.DataFrame = yf.download(tickers, start = period['start'], end = period['end'], progress=False)["Adj Close"]
    adjClosePrice.fillna(method = 'ffill', inplace=True)

    data: pd.DataFrame = pd.read_csv(f'./results/data/{tickers[0]}.csv', index_col='Date', parse_dates=True)
    label_w_ticker: str = f"period {data.index.min().strftime('%Y-%m-%d')} - {data.index.max().strftime('%Y-%m-%d')}"
    
    if use_ia:
        df_expected_return: pd.DataFrame = pd.read_csv('./results/prediction/Expected Return.csv', index_col='Ticker')
        aux_expected_return: Dict[str, float] = {}

        for ticker in tickers:
            aux_expected_return[ticker] = df_expected_return.loc[ticker, 'Expected Returns']

        forecast_return: pd.Series = pd.Series(aux_expected_return)
    
    else:
        forecast_return: pd.Series = expected_returns.mean_historical_return(adjClosePrice)
    
    cov_matrix: pd.DataFrame = risk_models.sample_cov(adjClosePrice)

    ef: EfficientFrontier = EfficientFrontier(forecast_return, cov_matrix)
    ef.add_constraint(lambda w: w >= 0)
    ef.add_constraint(lambda w: w <= maximum_participation)
    ef.max_sharpe(risk_free_rate = risk_free_rate)

    pre_pesos: Dict[str, float] = ef.clean_weights()
    pesos_restritos = {k: (v if v > minimum_participation else 0) for k, v in pre_pesos.items()}

    ef = EfficientFrontier(forecast_return, cov_matrix)
    ef.add_constraint(lambda w: w >= 0)
    ef.add_constraint(lambda w: w <= maximum_participation)
    ef.set_weights(pesos_restritos)

    performance: Tuple[float, float, float] = ef.portfolio_performance(verbose = False)
    pesos_limpos = ef.clean_weights() 

    pesos: pd.Series = pd.Series(pesos_limpos)

    test: pd.DataFrame = pd.DataFrame(columns=['Peso'])
    test.index.name = 'Ticker'
    for ticker, peso in zip(tickers, pesos):
        if peso > 0:
            test.loc[ticker, 'Peso'] = peso
        
    if not os.path.exists(f'./results/portfolio'):
        os.makedirs(f'./results/portfolio')

    test.to_csv('./results/portfolio/pesos.csv')

    pesos: pd.Series = pesos[pesos >= minimum_participation]
    explode: List[float] = [0.1 for _ in range(len(pesos))]


    text: str = f'Expected annual return: {performance[0]:.1%} \
                  \nAnnual volatility: {performance[1]:.1%} \
                  \nSharpe Ratio: {performance[2]:.2f}'

    percentuais: List[float] = [f'{p:.1f}%' for p in pesos.values * 100]
    legend_labels: List[str] = [f'{index[:-3]} - {percentual}' for index, percentual in zip(pesos.index, percentuais)]

    fig, ax = plt.subplots(figsize=(10, 5))
    wedges, texts, autotexts = ax.pie(pesos.values, explode=explode, labels=pesos.index, autopct='%1.1f%%',
                                      shadow=True, startangle=90)
    
    ax.legend(wedges, legend_labels, title="Ações", loc="upper left", bbox_to_anchor=(1.2, 0, 0.5, 1))
    ax.set_title(f"Portifólio Otimizado - {label_w_ticker}")
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.75, 0.15, text, wrap=True, ha='left', va='top', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'./results/portfolio/Portifólio Otimizado - {label_w_ticker}.png')
    plt.close()

    ef = EfficientFrontier(forecast_return, cov_matrix)
    _, ax = plt.subplots(figsize=(10, 5))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True, risk_free_rate = risk_free_rate, market_map=True)
    plt.title(f'Fronteira Eficiente de Markowitz - {label_w_ticker}')
    plt.savefig(f'./results/portfolio/Fronteira de Eficiência - {label_w_ticker}.png')
    plt.close()

if __name__ == '__main__':

    from variables import tickers, period, SEED, risk_free_rate, minimum_participation, maximum_participation

    make_portfolio(tickers,
                   period,
                   SEED,
                   risk_free_rate,
                   minimum_participation,
                   maximum_participation,
                   use_ia = False
                   )