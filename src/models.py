# Keras with tensorflow 1.x backend

import keras.metrics
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras import layers as kl
from keras.layers.merge import concatenate
from keras import Model

# Single branch Sequential model
def build_basic_model(model_name, window, train_shape, loss, dropout=0.2):
    dropout = dropout    
    I = Input(shape=(window, train_shape), name="basic")    
    
    if model_name == "LSTM":  
        X = LSTM(20, return_sequences=True)(I)
        X = LSTM(20, return_sequences=False)(X)    
        
    elif model_name == "RNN":
        X = SimpleRNN(20, return_sequences=True)(I) 
        X = SimpleRNN(20, return_sequences=False)(X) 
      
    elif model_name == "GRU":
        X = GRU(20, return_sequences=True)(I) 
        X = GRU(20, return_sequences=False)(X) 

    elif model_name == "BLSTM":
        X = kl.Bidirectional(LSTM(20, return_sequences=True))(I) 
        X = kl.Bidirectional(LSTM(20, return_sequences=False))(X) 
      
    elif model_name == "BGRU":
        X = kl.Bidirectional(GRU(20, return_sequences=True))(I) 
        X = kl.Bidirectional(GRU(20, return_sequences=False))(X) 
       
    elif model_name == "BRNN":
        X = kl.Bidirectional(SimpleRNN(20, return_sequences=True))(I) 
        X = kl.Bidirectional(SimpleRNN(20, return_sequences=False))(X) 
        
    X = Dropout(dropout)(X)
    X = Dense(20, activation="relu")(X)
    Y = Dense(1, activation='linear')(X)

    model = Model(inputs=I, outputs=Y)

    model.compile(loss=loss, optimizer='adam', metrics=[keras.metrics.mape])

    return model

#Branched sequential model to process basic and derived features seperately
def build_deep_parse_model(model_name, window, basic_shape, derived_shape, loss, dropout=0.2):

    I1 = Input(shape=(window, basic_shape), name="basic")
    I2 = Input(shape=(window, derived_shape), name="derived")
    

    if model_name == "BLSTM":    
        X1 = kl.Bidirectional(LSTM(20, return_sequences=True))(I1)
        X1 = kl.Bidirectional(LSTM(20, return_sequences=False))(X1)

        X2 = kl.Bidirectional(LSTM(20,  return_sequences=True))(I2)
        X2 = kl.Bidirectional(LSTM(20, return_sequences=False))(X2)

    elif model_name == "BGRU":    
        X1 = kl.Bidirectional(GRU(20, return_sequences=True))(I1)
        X1 = kl.Bidirectional(GRU(20, return_sequences=False))(X1)

        X2 = kl.Bidirectional(GRU(20,  return_sequences=True))(I2)
        X2 = kl.Bidirectional(GRU(20, return_sequences=False))(X2)

    elif model_name == "BRNN":
        X1 = kl.Bidirectional(SimpleRNN(20, return_sequences=True))(I1)
        X1 = kl.Bidirectional(SimpleRNN(20, return_sequences=False))(X1)

        X2 = kl.Bidirectional(SimpleRNN(20,  return_sequences=True))(I2)
        X2 = kl.Bidirectional(SimpleRNN(20, return_sequences=False))(X2)

    elif model_name == "LSTM":
        X1 = (LSTM(20, return_sequences=True))(I1)
        X1 = (LSTM(20, return_sequences=False))(X1)

        X2 = (LSTM(20,  return_sequences=True))(I2)
        X2 = (LSTM(20, return_sequences=False))(X2)

    elif model_name == "GRU":
        X1 = (GRU(20, return_sequences=True))(I1)
        X1 = (GRU(20, return_sequences=False))(X1)

        X2 = (GRU(20,  return_sequences=True))(I2)
        X2 = (GRU(20, return_sequences=False))(X2)

    elif model_name == "RNN":  
        X1 = (SimpleRNN(20, return_sequences=True))(I1)
        X1 = (SimpleRNN(20, return_sequences=False))(X1)

        X2 = (SimpleRNN(20,  return_sequences=True))(I2)
        X2 = (SimpleRNN(20, return_sequences=False))(X2)
       

    X1 = Dropout(dropout)(X1)
    X2 = Dropout(dropout)(X2)

    X = concatenate([X1, X2], axis=-1)
    X = Dense(20, activation="relu")(X)
    Y = Dense(1, activation="linear")(X)
    model = Model(inputs=[I1, I2], outputs=Y)

    model.compile(loss=loss, optimizer='adam', metrics=[keras.metrics.mape])
    
    return model
