import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import yfinance as yf
from pypfopt import EfficientFrontier, EfficientCVaR, expected_returns, risk_models, plotting

plt.style.use('ggplot')
plt.rc('figure', autolayout=True)

def optimizer(tickers, period, SEED, risk_free_rate, minimum_participation, use_ia = True):

    random.seed(SEED)
    np.random.seed(SEED)

    adjClosePrice = yf.download(tickers, start = period['start'], end = period['end'])["Adj Close"]
    adjClosePrice = adjClosePrice.fillna(method = 'ffill')

    data = pd.read_csv(f'./results/data/{tickers[0]}.csv', index_col='Date', parse_dates=True)  

    label_w_ticker = f"trained: {data.index.min().strftime('%Y-%m-%d')} -> {data.index.max().strftime('%Y-%m-%d')}"
    
    if use_ia:
        df_expected_return = pd.read_csv('./results/prediction/Expected Return.csv', index_col='Ticker')
        aux_expected_return = dict()

        for ticker in tickers:
            aux_expected_return[ticker] = df_expected_return.loc[ticker, 'Expected Returns']

        forecast_return = pd.Series(aux_expected_return)
    
    else:
        forecast_return = expected_returns.mean_historical_return(adjClosePrice)
    
    cov_matrix = risk_models.sample_cov(adjClosePrice)

    ef = EfficientFrontier(forecast_return, cov_matrix)
    # ef = EfficientCVaR(forecast_return, cov_matrix)

    ef.add_constraint(lambda w: w >= 0)

    ef.max_sharpe(risk_free_rate = risk_free_rate)
    #pesos_brutos = ef.min_cvar()

    pesos_limpos = ef.clean_weights()
    performance = ef.portfolio_performance(verbose = False)

    # pesos_restritos = {k: (v if v > 0.05 else 0) for k, v in pesos_limpos.items()}
    # ef = EfficientFrontier(forecast_return, cov_matrix)
    # ef.set_weights(pesos_restritos)
    # ef.max_sharpe()
    # pesos_finais = ef.clean_weights()

    pesos = pd.Series(pesos_limpos)

    outros = pesos[pesos < minimum_participation].sum()
    pesos_pie = pesos[pesos >= 0.05]

    explode = [0.1 for _ in range(len(pesos_pie))]

    if outros > 0:
        pesos_pie['Outros'] = outros
        explode.append(0.1)

    # print(performance)

    text: str = f'Expected annual return: {performance[0]:.1%} \
                  \nAnnual volatility: {performance[1]:.1%} \
                  \nSharpe Ratio: {performance[2]:.2f}'
    

    if not os.path.exists(f'./results/portfolio'):
        os.makedirs(f'./results/portfolio')

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(pesos_pie.values, explode=explode, labels=pesos_pie.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
    
    # Criando rótulos personalizados para a legenda que incluem o percentual
    percentuais = [f'{p:.1f}%' for p in pesos_pie.values / pesos_pie.values.sum() * 100]
    legend_labels = [f'{index} - {percentual}' for index, percentual in zip(pesos_pie.index, percentuais)]
    
    ax.legend(wedges, legend_labels, title="Assets", loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title("Portifólio Otimizado")
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.6, 0.15, text, wrap=True, ha='left', va='top', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'./results/portfolio/Portifólio Otimizado - {label_w_ticker}.png')
    plt.close()

    
    ef = EfficientFrontier(forecast_return, cov_matrix)
    _, ax = plt.subplots()
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True, risk_free_rate = risk_free_rate, market_map=True, cmap='viridis') #
    plt.title('Fronteira Eficiente de Markowitz')
    plt.savefig(f'./results/portfolio/Fronteira de Eficiência - {label_w_ticker}.png')
    plt.close()

if __name__ == '__main__':

    from variables import tickers, period, SEED, risk_free_rate, minimum_participation

    optimizer(tickers, period, SEED, risk_free_rate, minimum_participation, use_ia = True)

