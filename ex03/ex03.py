import keras
from keras.layers import Dense

model = keras.Sequential()
model.add(Dense(8, input_shape=(5,), activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='linear'))  # linear for regression output

print(model.summary())
