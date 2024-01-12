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

    label_w_ticker: str = f"period {adjClosePrice.index.min().strftime('%Y-%m-%d')} - {adjClosePrice.index.max().strftime('%Y-%m-%d')}"
    if use_ia:
        df_expected_return: pd.DataFrame = pd.read_csv('./results/prediction/Expected Return.csv', index_col='Ticker')
        aux_expected_return: Dict[str, float] = {}

        for ticker in tickers:
            aux_expected_return[ticker] = df_expected_return.loc[ticker, 'Expected Returns']

        forecast_return: pd.Series = pd.Series(aux_expected_return)
    
    else:
        forecast_return: pd.Series = expected_returns.mean_historical_return(adjClosePrice)
    
    cov_matrix: pd.DataFrame = risk_models.sample_cov(adjClosePrice)

    pre_ef: EfficientFrontier = EfficientFrontier(forecast_return, cov_matrix, solver='CLARABEL') # ['CLARABEL', 'ECOS', 'ECOS_BB', 'OSQP', 'SCIPY', 'SCS']
    pre_ef.add_constraint(lambda w: w >= 0)
    pre_ef.add_constraint(lambda w: w <= maximum_participation)

    try:
        pre_ef.max_sharpe(risk_free_rate = risk_free_rate) # maximal Sharpe ratio (a.k.a the tangency portfolio)
        # pre_ef.min_volatility() # minimum volatility
        # pre_ef.max_quadratic_utility() # maximises the quadratic utility, given some risk aversion.
        # pre_ef.efficient_risk(target_volatility=0.3) # maximises return for a given target risk
        # pre_ef.efficient_return(target_return=0.4) # minimises risk for a given target return

    except Exception as e:
        print(f"\nLimites e Restrições Inalcançáveis! Verifique a integridade dos dados baixados em ./src/data ou tente fornecer mais opções de ações\n")
        return

    raw_weights: Dict[str, float] = pre_ef.clean_weights()
    restrict_weights = {k: (v if v > minimum_participation else 0) for k, v in raw_weights.items()}

    ef = EfficientFrontier(forecast_return, cov_matrix)
    ef.add_constraint(lambda w: w <= maximum_participation)
    ef.set_weights(restrict_weights)

    performance: Tuple[float, float, float] = ef.portfolio_performance(verbose = False)
    pesos_limpos = ef.clean_weights() 

    pesos: pd.Series = pd.Series(pesos_limpos)
    pesos = pesos[pesos >= minimum_participation]

    test: pd.DataFrame = pd.DataFrame(columns=['Peso'])
    test.index.name = 'Ticker'
    for ticker, peso in zip(tickers, pesos):
        if peso > 0:
            test.loc[ticker, 'Peso'] = peso
        
    if not os.path.exists(f'./results/portfolio'):
        os.makedirs(f'./results/portfolio')

    test.to_csv('./results/portfolio/weight.csv')

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
    plt.savefig(f'./results/portfolio/Portifólio Otimizado - {label_w_ticker}.png')
    plt.close()

    ef = EfficientFrontier(forecast_return, cov_matrix)
    _, ax = plt.subplots(figsize=(10, 5))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True, risk_free_rate = risk_free_rate)
    plt.title(f'Fronteira Eficiente de Markowitz - {label_w_ticker}')
    plt.plot(performance[1], performance[0], 'bo', label=f'Portifólio Otimizado - Sharpe Ratio: {performance[2]:.2f} %')
    plt.legend(loc='best')
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