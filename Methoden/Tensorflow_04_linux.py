# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:28:00 2024
@author: kesper
"""
# Import der notwendigen Module
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
 
path = "/mnt/c/Users/Admin/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Train.csv"

data = pd.read_csv(path, sep=';', decimal='.')

x = data[['room_type_encoded', 'rest_index_norm', 'metro_dist', 'dist', 'bedrooms', 'AttractionScore_Norm', 'city_encoded',]].values
y = data[['realSum_Normalized']].values
 
# Erstellen eines einfachen neuronalen Netzwerks
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(7,)),  # Eingabe ist 7-Dimensional
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Die Ausgabe ist 1-dimensional
])
 
# Kompilieren des Modells
model.compile(optimizer='adam', loss='mean_squared_error')  # Für Regressionsaufgaben
 
# Training des Modells
model.fit(x, y, epochs=15000)  # Wir verwenden die gleichen Daten für Training und Test
 
# Evaluieren des Modells (optional)
loss = model.evaluate(x, y)
print(f'Loss auf den Trainingsdaten: {loss:.4f}')
 
# Vorhersagen mit dem trainierten Modell
predictions = model.predict(x)
print(predictions)
model.save("/mnt/c/Users/Admin/FOM-Anwendungsprojekt/Models/train-csv-r_t_e-r_i_n-m_d-d-b-AS_N-c_e-15kepochs_001.h5")

#testpath = "/Users/MK/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv"
#testdata = pd.read_csv(testpath, sep=';', decimal='.')
#x_t = testdata[['cleanliness_rating', 'guest_satisfaction_overall', 'AttractionScore_Norm']].values
#y_t = testdata[['realSum']].values
#predictions = model.predict(x_t)
#model.save("/mnt/c/Users/Admin/FOM-Anwendungsprojekt/Models/c_rt-g_s_o-as_n-100epochs_002.h5")