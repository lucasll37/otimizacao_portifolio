import warnings
warnings.filterwarnings('ignore')

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, TensorBoard
from typing import List


def create_callbacks(label: str, optim: bool = False, verbose: int = 1) -> List[Callback]:

    
    earlystop: Callback = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=15,
                            verbose=verbose,
                            restore_best_weights=True)

    reduceLr: Callback = ReduceLROnPlateau(monitor='loss',
                                factor=0.2,
                                patience=5,
                                mode="min",
                                verbose=verbose,
                                min_delta=0.0001,
                                min_lr=0)
    
    tensorboard: Callback = TensorBoard(log_dir=f'./results/logs/{label}', histogram_freq=1)


    callbacks: List[Callback] = [earlystop, reduceLr, tensorboard, TerminateOnNaN()]
    callbacks_w_logs: List[Callback] = [earlystop, reduceLr, TerminateOnNaN()]
    
    return callbacks_w_logs if optim else callbacks