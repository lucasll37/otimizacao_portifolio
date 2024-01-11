import warnings
warnings.filterwarnings('ignore')

from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Model, Sequential

def create_model(optim: str,
                 layers: int,
                 n_lstm: int,
                 dropoutFoward: int,
                 stepsBack: int,
                 stepsFoward: int,
                 ) -> Model:

    model: Model = Sequential()
    
    model.add(Input(shape=(stepsBack, 1)))
    
    #################################################################
    for _ in range(layers-1):
        model.add(LSTM(n_lstm,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=True))
        
        model.add(Dropout(dropoutFoward))
    ##################################################################
    
    model.add(LSTM(n_lstm,
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   return_sequences=False))

    model.add(Dense(stepsFoward, activation='linear'))

    # model.compile(loss='mean_squared_error', optimizer=optim)
    model.compile(loss='mean_absolute_percentage_error', optimizer=optim)

    return model

if __name__ == '__main__':

    model: Model = create_model(optim='Adagrad',
                                layers=3,
                                n_lstm=240,
                                dropoutFoward=0.05,
                                stepsBack=240,
                                stepsFoward=60
                                )
    
    model.summary()