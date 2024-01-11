from warnings import filterwarnings
filterwarnings('ignore')

import sys
from src.obtaining import getData
from src.callbacks import create_callbacks
from src.model import create_model
from src.optimizer import optimizer
from src.train import train
from src.prediction import prediction
from src.monteCarlo import monteCarlo
from src.portfolio import make_portfolio

from src.variables import tickers, period, observation_window, monte_carlo_simulation, \
                          SEED, n_trials_optuna, epochs, risk_free_rate, verbose, \
                          minimum_participation, maximum_participation


def main():
    if "--no-download" not in sys.argv:
        for ticker in tickers:
            getData(ticker, period)
            
    if "--only-download" in sys.argv:
        return

    if "--no-ai" not in sys.argv:
        
        if "--no-train" not in sys.argv:
            train(
                create_model,
                create_callbacks,
                optimizer,
                tickers,
                period,
                observation_window,
                SEED,
                n_trials_optuna,
                epochs,
                graphics=True,
                verbose=verbose
            )

        prediction(tickers, period, SEED, observation_window, graphics=True)
        
        monteCarlo(
            tickers,
            period,
            observation_window,
            monte_carlo_simulation,
            SEED,
            graphics=True
        )

        make_portfolio(tickers,
                       period,
                       SEED,
                       risk_free_rate,
                       minimum_participation,
                       maximum_participation,
                       use_ia = True)
    
    else:
        make_portfolio(tickers,
                       period,
                       SEED,
                       risk_free_rate,
                       minimum_participation, 
                       maximum_participation,
                       use_ia = False
                       )

if __name__ == '__main__':
    main()
