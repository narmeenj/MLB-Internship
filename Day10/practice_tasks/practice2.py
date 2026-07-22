import os      
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import tensorflow as tf 

from keras.models import Sequential
from keras.layers import Dense, Input

model =Sequential()

#input layer
model.add(Input(shape=(4,)))

#hidden layer
model.add(Dense(units=16,activation='relu', name='Hidden_Layer'))

#output layer
model.add(Dense(units=1,activation='sigmoid', name='Output_Layer'))

model.summary()


