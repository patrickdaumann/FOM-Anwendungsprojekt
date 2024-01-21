# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 14:10:23 2024
@author: MK
"""
# Import der notwendigen Module
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt


path = "/mnt/c/Users/MK/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Train.csv"
#path = "/Users/patrick/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Train.csv"
val_path = "/mnt/c/Users/MK/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Val.csv"
#val_path = "/Users/patrick/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Val.csv"

data = pd.read_csv(path, sep=';', decimal='.')

x = data[['room_type_encoded', 'rest_index_norm', 'metro_dist', 'dist', 'bedrooms', 'AttractionScore_Norm', 'city_encoded',]].values
y = data[['realSum_Normalized']].values


val_data = pd.read_csv(path, sep=';', decimal='.')

val_x = val_data[['room_type_encoded', 'rest_index_norm', 'metro_dist', 'dist', 'bedrooms', 'AttractionScore_Norm', 'city_encoded',]].values
val_y = val_data[['realSum_Normalized']].values

print(val_x)
print(val_y)

 
# Erstellen eines einfachen neuronalen Netzwerks
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(7,)),  # Eingabe ist 7-Dimensional
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Die Ausgabe ist 1-dimensional
])
 
# Kompilieren des Modells
model.compile(optimizer='adam', loss='mean_squared_error')  # Für Regressionsaufgaben
 
# Training des Modells
#model.fit(x, y, epochs=15000)  # Wir verwenden die gleichen Daten für Training und Test
 
history = model.fit(x, y, epochs=15000, validation_data=(val_x, val_y))

# Visualisierung Plot
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Evaluieren des Modells
loss = model.evaluate(x, y)
print(f'Loss auf den Trainingsdaten: {loss:.4f}')
 
# Vorhersagen mit dem trainierten Modell
predictions = model.predict(x)
print(predictions)
model.save("/mnt/c/Users/Admin/FOM-Anwendungsprojekt/Models/train-csv-r_t_e-r_i_n-m_d-d-b-AS_N-c_e-15kepochs_002-moreLayers.h5")
 

#testpath = "/Users/MK/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv"
#testdata = pd.read_csv(testpath, sep=';', decimal='.')
#x_t = testdata[['cleanliness_rating', 'guest_satisfaction_overall', 'AttractionScore_Norm']].values
#y_t = testdata[['realSum']].values
#predictions = model.predict(x_t)
#model.save("/mnt/c/Users/Admin/FOM-Anwendungsprojekt/Models/c_rt-g_s_o-as_n-100epochs_002.h5")