import warnings
warnings.filterwarnings('ignore')

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, TensorBoard


def create_callbacks(ticker, label, optim = False, verbose=1):
    
    earlystop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=7,
                            verbose=verbose,
                            restore_best_weights=True)

    reduceLr = ReduceLROnPlateau(monitor='loss',
                                factor=0.2,
                                patience=3,
                                mode="min",
                                verbose=verbose,
                                min_delta=0.0001,
                                min_lr=0)
    
    tensorboard = TensorBoard(log_dir=f'./results/logs/{ticker}/{label}', histogram_freq=1)


    callbacks = [earlystop, reduceLr, tensorboard, TerminateOnNaN()]
    callbacks_w_logs = [earlystop, reduceLr, TerminateOnNaN()]
    
    return callbacks_w_logs if optim else callbacks