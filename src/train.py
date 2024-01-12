import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import optuna
import os
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.models import Model
from keras.callbacks import History

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump
from typing import Callable, Dict, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [10, 5]
plt.rc('figure', autolayout=True)


def train(
        create_model: Callable,
        create_callbacks: Callable,
        optimizer: Callable,
        tickers: List[str],
        period: Dict[str, str],
        observation_window: Dict[str, int],
        SEED: int,
        n_trials_optuna: int,
        epochs: int,
        graphics: bool = True,
        verbose: int = 1
) -> None:

    random.seed(SEED)
    np.random.seed(SEED)
    
    for ticker in tickers:
        print(f"training predictive model for the asset {ticker}...")

        try:
            data: pd.DataFrame = pd.read_csv(f'./results/data/{ticker}.csv', index_col='Date', parse_dates=True)
            label: str = f"{ticker} period {data.index.min().strftime('%Y-%m-%d')} - {data.index.max().strftime('%Y-%m-%d')}"
            
            df_train_valid: pd.DataFrame = data[data.index < pd.Timestamp(period['boundary'])][['Adj Close']]
            df_test: pd.DataFrame = data[data.index >= pd.Timestamp(period['boundary'])][['Adj Close']]

            scaler: MinMaxScaler = MinMaxScaler()

            scaled_df_train_valid: np.ndarray = scaler.fit_transform(df_train_valid.dropna())
            scaled_df_test: np.ndarray = scaler.transform(df_test.dropna())

            if not os.path.exists(f'./results/serialized objects/{ticker}'):
                os.makedirs(f'./results/serialized objects/{ticker}')
            
            dump(scaler, f'./results/serialized objects/{ticker}/scaler - {label}.joblib')
            
            
            if graphics:
                plt.title(f"Divisão de data Treino/Validação - {ticker}")
                plt.plot(df_train_valid.index, df_train_valid['Adj Close'], label = 'Dados de Treino e Validação') 
                plt.plot(df_test.index, df_test['Adj Close'], label = 'Dados de Teste') 
                plt.xticks(rotation=45)
                plt.legend(loc = 'best')

                if not os.path.exists(f'./results/graphics/{ticker}/{label}/Training'):
                    os.makedirs(f'./results/graphics/{ticker}/{label}/Training')

                plt.savefig(f'./results/graphics/{ticker}/{label}/Training/Divisão de data Treino-Validação.png')
                plt.close()


            X_train_valid: List[np.ndarray] = []
            y_train_valid: List[np.ndarray] = []

            for i in range(observation_window['stepsBack'], len(scaled_df_train_valid) - observation_window['stepsFoward']):
                X_train_valid.append(scaled_df_train_valid[i - observation_window['stepsBack']:i, 0])
                y_train_valid.append(scaled_df_train_valid[i:i + observation_window['stepsFoward'], 0])

            X_train_valid: np.ndarray = np.array(X_train_valid)
            y_train_valid: np.ndarray = np.array(y_train_valid)
            X_train_valid: np.ndarray = np.reshape(X_train_valid, (X_train_valid.shape[0], X_train_valid.shape[1], 1))

            X_train: np.ndarray
            X_valid: np.ndarray
            y_train: np.ndarray
            y_valid: np.ndarray
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train_valid,
                y_train_valid,
                test_size = 0.2,
                shuffle = True,
                random_state = SEED
            )

            X_test: List[np.ndarray] = []
            y_test: List[np.ndarray] = []

            for i in range(observation_window['stepsBack'], len(scaled_df_test) - observation_window['stepsFoward']):
                X_test.append(scaled_df_test[i - observation_window['stepsBack']:i, 0])
                y_test.append(scaled_df_test[i:i + observation_window['stepsFoward'], 0])

            X_test: np.ndarray = np.array(X_test)
            y_test: np.ndarray = np.array(y_test)
            X_test: np.ndarray = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            if not os.path.exists(f'./sqlite/{ticker}'):
                os.makedirs(f'./sqlite/{ticker}')

            study: optuna.study.Study = optuna.create_study(
                study_name=label,
                storage=f"sqlite:///./sqlite/{ticker}/{label}.db",
                load_if_exists=True,
                direction='minimize'
            )


            study.optimize(
                lambda trial: optimizer(
                    trial,
                    create_model,
                    create_callbacks,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    ticker,
                    label,
                    epochs,
                    observation_window,
                    verbose
                ),
                n_trials = n_trials_optuna
            )

            model: Model = create_model(
                optim = study.best_params['optim'],
                layers = study.best_params['layers'],
                n_lstm = study.best_params['n_lstm'],
                dropoutFoward = study.best_params['dropoutFoward'],
                stepsBack = observation_window['stepsBack'],
                stepsFoward = observation_window['stepsFoward']
            )

            history: History = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=study.best_params['batch_size'],
                validation_data = (X_valid, y_valid),
                callbacks = create_callbacks(ticker, label, False, verbose)
            )

            model.save(f'./results/trained models/{ticker}/{label}.h5')

            if graphics:
                for i in range(240, len(scaled_df_test) - observation_window['stepsFoward']):
                    X_temp: np.ndarray = scaled_df_test[i - observation_window['stepsBack']:i, 0]
                    y_temp: np.ndarray = scaled_df_test[i:i + observation_window['stepsFoward'], 0]

                    scaled_return_forecast: np.ndarray = model.predict(X_temp.reshape(1, -1, 1))

                    return_forecast: np.ndarray = scaler.inverse_transform(scaled_return_forecast)
                    return_stepback: np.ndarray = scaler.inverse_transform(X_temp.reshape(-1, 1))
                    _return: np.ndarray = scaler.inverse_transform(y_temp.reshape(-1, 1))
                    
                    pd_return_stepback: pd.Series = pd.Series(
                        return_stepback.reshape(-1),
                        index=df_test.index[i - observation_window['stepsBack']:i])
                    
                    pd_return_forecast: pd.Series = pd.Series(
                        return_forecast.reshape(-1),
                        index=df_test.index[i:i + observation_window['stepsFoward']])
                    
                    pd_return: pd.Series = pd.Series(
                        _return.reshape(-1),
                        index=df_test.index[i:i + observation_window['stepsFoward']])
                    
                    # Plot
                    plt.title(f"Previsão de {ticker} - {pd_return_stepback.index[-1].strftime('%Y-%m-%d')}")
                    plt.plot(pd_return_stepback, label = 'Adj Close (SteBack)') 
                    plt.plot(pd_return_forecast, label = 'Adj Close Forecast') 
                    plt.plot(pd_return, color = 'orange', label = 'Adj Close') 
                    plt.xticks(rotation=45)
                    plt.legend(loc = 'best')
                                
                    if not os.path.exists(f'./results/graphics/{ticker}/{label}/Training'):
                        os.makedirs(f'./results/graphics/{ticker}/{label}/Training')
                    
                    plt.savefig(f'./results/graphics/{ticker}/{label}/Training/{pd_return_stepback.index[-1].strftime("%Y-%m-%d")}.png')
                    plt.close()

        except Exception as e:
            print(f"Error when training predictive model for the asset {ticker}. Detail: {e}")



if __name__ == '__main__':

    from model import create_model
    from callbacks import create_callbacks
    from optimizer import optimizer
    from variables import tickers, period, observation_window, SEED, n_trials_optuna, epochs, verbose
    
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
