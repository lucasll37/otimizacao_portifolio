import warnings
warnings.filterwarnings('ignore')

from keras.models import Model
from keras.callbacks import History
from typing import Callable, Dict
import numpy as np
import optuna

def optimizer(
        trial: optuna.Trial,
        create_model: Callable,
        create_callbacks: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        ticker: str,
        label: str,
        epochs: int,
        observation_window: Dict[str, int],
        verbose: int
    ) -> float:

    n_base_lstm: int = observation_window['stepsBack']
    
    optim: str = trial.suggest_categorical('optim', ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adamax', 'Adagrad'])
    layers: int = trial.suggest_categorical('layers', [2, 3, 4])
    n_lstm: int = trial.suggest_categorical('n_lstm', [120, 240, 360, 480])
    batch_size: int = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropoutFoward: float = trial.suggest_categorical('dropoutFoward', [0, 0.05]) 
    
    modelStudy: Model = create_model(
        optim=optim,
        layers=layers,
        n_lstm=n_lstm,
        dropoutFoward=dropoutFoward,
        stepsBack=observation_window['stepsBack'],
        stepsFoward=observation_window['stepsFoward']
    )

    history: History = modelStudy.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_valid, y_valid),
        callbacks=create_callbacks(label, True, verbose)
    )
    
    return min(history.history['val_loss'])