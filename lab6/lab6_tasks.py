# DO NOT IMPORT ANY ADDITIONAL LIBRARY! (e.g. sklearn)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def shuffle_data_numpy(X, y, numpy_seed):
    # fix the random seed
    np.random.seed(numpy_seed)

    # TODO Task 1.1
    # shuffle the given data pair (X, y)
    # please use numpy functions so that the results are controled by np.random.seed(numpy_seed)
    indexes=np.arange(X.shape[0])
    np.random.shuffle(indexes)
    X_shuffle=X.take(indexes,axis=0)
    y_shuffle=y.take(indexes,axis=0)
    #np.random.shuffle(X)
    #np.random.seed(numpy_seed)
    #np.random.shuffle(y)
    return X_shuffle, y_shuffle

def train_val_split(X_trainval, y_trainval, train_size, numpy_seed):
   # TODO TASK 1.2 
    # apply shuffle on the data with given random seed, then split the data into training and validation sets
    X_shuffle,y_shuffle=shuffle_data_numpy(X_trainval,y_trainval,numpy_seed)
    train_indexes=np.arange(train_size)
    validation_indexes=np.arange(train_size,X_trainval.shape[0])
    X_train=np.take(X_shuffle,train_indexes,axis=0)
    y_train=np.take(y_shuffle,train_indexes,axis=0)
    X_val=np.take(X_shuffle,validation_indexes,axis=0)
    y_val=np.take(y_shuffle,validation_indexes,axis=0)
    return X_train, X_val, y_train, y_val

def MyModel(num_dense_layer, dense_layer_unit, input_dim, dropout_ratio):
    # Create a sequential model
    model = Sequential()

    # TODO Task 2.1
    # Build your own model with model.add(), Dense layers, and Dropout layers
    # Hint: you may consider using
        # Dense(): https://keras.io/api/layers/core_layers/dense/
        # Dropout(): https://keras.io/api/layers/regularization_layers/dropout/
    model.add(Dense(units=dense_layer_unit,input_dim=input_dim,activation='relu',kernel_initializer='uniform'))
    model.add(Dropout(rate=dropout_ratio))
    for _ in range(num_dense_layer-1):
        model.add(Dense(units=dense_layer_unit,kernel_initializer='uniform',activation='relu'))
        model.add(Dropout(rate=dropout_ratio))
    model.add(Dense(units=1,kernel_initializer='uniform',activation='linear'))
    return model

def MyModel_Training(model, X_train, y_train, X_val, y_val, batchsize, train_epoch):

    # TODO Task 2.2
    # Compile and train the given model
    # Hint: history can be returned by model.fit() function, please see https://keras.io/api/models/model_training_apis/
    adam_optimizer=Adam(learning_rate=1e-3)
    model.compile(optimizer=adam_optimizer,loss='mse',metrics=['mae'])
    history=model.fit(X_train,y_train,epochs=train_epoch,validation_data=(X_val,y_val))
    return history, model