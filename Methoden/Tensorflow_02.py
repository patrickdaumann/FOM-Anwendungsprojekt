# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Import der notwendigen Module
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
 
path = "/Users/MK/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Full.csv"

data = pd.read_csv(path, sep=';', decimal='.')

x = data[['cleanliness_rating', 'guest_satisfaction_overall', 'AttractionScore_Norm']].values
y = data[['realSum']].values
 
# Erstellen eines einfachen neuronalen Netzwerks
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(3,)),  # Eingabe ist 3-dimensional
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Die Ausgabe ist 1-dimensional
])
 
# Kompilieren des Modells
model.compile(optimizer='adam', loss='mean_squared_error')  # Für Regressionsaufgaben
 
# Training des Modells
model.fit(x, y, epochs=100)  # Wir verwenden die gleichen Daten für Training und Test
 
# Evaluieren des Modells (optional)
loss = model.evaluate(x, y)
print(f'Loss auf den Trainingsdaten: {loss:.4f}')
 
# Vorhersagen mit dem trainierten Modell
predictions = model.predict(x)
print(predictions)
 
#testpath = "/Users/MK/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv"
#testdata = pd.read_csv(testpath, sep=';', decimal='.')
#x_t = testdata[['cleanliness_rating', 'guest_satisfaction_overall', 'AttractionScore_Norm']].values
#y_t = testdata[['realSum']].values
#predictions = model.predict(x_t)
#model.save("/Users/MK/Documents/GitHub/FOM-Anwendungsprojekt/Models/c_rt-g_s_o-as_n_100epochs_001.h5")