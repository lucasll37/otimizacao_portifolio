from warnings import filterwarnings
filterwarnings('ignore')

from src.obtaining import getData
from src.callbacks import create_callbacks
from src.model import create_model
from src.train import train
from src.prediction import prediction
from src.monteCarlo import monteCarlo
from src.portfolio import optimizer

from src.variables import tickers, period, observation_window, monte_carlo_simulation, \
                          SEED, n_trials_optuna, epochs, risk_free_rate, verbose, \
                          minimum_participation


def main():

    for ticker in tickers:
        getData(ticker, period)

    train(
        create_model,
        create_callbacks,
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
    
    optimizer(tickers, period, SEED, risk_free_rate, minimum_participation)

if __name__ == '__main__':
    main()
