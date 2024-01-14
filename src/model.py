import warnings
warnings.filterwarnings('ignore')

from keras.layers import Input, LSTM, Dropout, Dense # , Transformer
from keras.models import Model, Sequential

def create_model(optim: str,
                 layers: int,
                #  n_transformer: int,
                #  dropoutForward: float,
                 n_lstm: int, #
                 dropoutFoward: int, #
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
        

    # for _ in range(layers-1):
    #     model.add(Transformer(n_transformer,
    #                           feed_forward_dim=n_transformer * 4,
    #                           num_heads=4,
    #                           dropout_rate=dropoutForward,
    #                           return_sequences=True if _ < layers - 1 else False))
        
        model.add(Dropout(dropoutFoward))
    ##################################################################

    
    model.add(LSTM(n_lstm,
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   return_sequences=False))
    
    # model.add(Transformer(n_transformer,
    #                       feed_forward_dim=n_transformer * 4,
    #                       num_heads=4,
    #                       dropout_rate=dropoutForward,
    #                       return_sequences= False))

    model.add(Dense(stepsFoward, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer=optim)

    return model

if __name__ == '__main__':

    from variables import observation_window

    model: Model = create_model(optim='Adagrad',
                                layers=3,
                                # n_transformer: int,
                                # dropoutForward: float,
                                n_lstm=observation_window['stepsBack'],
                                dropoutFoward=0.05,
                                stepsBack=observation_window['stepsBack'],
                                stepsFoward=observation_window['stepsFoward']
                                )
    
    model.summary()