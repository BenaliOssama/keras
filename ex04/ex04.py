import keras
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Exercise 4.1: Train without scaling
print("=" * 50)
print("Exercise 4.1: Without Scaling")
print("=" * 50)

model = keras.Sequential()
model.add(Dense(10, input_shape=(30,), activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=300, verbose=0)

print(f"Final accuracy: {model.history.history['accuracy'][-1]:.4f}")

# Exercise 4.2: Train with scaling
print("\n" + "=" * 50)
print("Exercise 4.2: With Scaling")
print("=" * 50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model2 = keras.Sequential()
model2.add(Dense(10, input_shape=(30,), activation='sigmoid'))
model2.add(Dense(5, activation='sigmoid'))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.fit(X_scaled, y, epochs=50, batch_size=300, verbose=0)

print(f"Final accuracy: {model2.history.history['accuracy'][-1]:.4f}")
