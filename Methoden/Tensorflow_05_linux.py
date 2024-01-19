# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:28:00 2024
@author: kesper
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
 
# Beispiel-Daten: Ein eindimensionales Array von Features und zugehörige Zielvariablen
#X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
#y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

# Annahme: Ihre Daten sind in einer Datei mit dem Namen "data.csv"
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
    layers.Dense(64, activation='relu', input_shape=(7,)),  # Eingabe ist 7-Dimensional
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Die Ausgabe ist 1-dimensional
])
 
# Kompilieren des Modells
model.compile(optimizer='adam', loss='mean_squared_error')  # Für Regressionsaufgaben
 
# Training des Modells
#model.fit(x, y, epochs=15000)  # Wir verwenden die gleichen Daten für Training und Test
 
history = model.fit(x, y, epochs=100, validation_data=(val_x, val_y))




plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()



# Evaluieren des Modells (optional)
loss = model.evaluate(x, y)
print(f'Loss auf den Trainingsdaten: {loss:.4f}')
 
# Vorhersagen mit dem trainierten Modell
predictions = model.predict(x)
print(predictions)
model.save("/mnt/c/Users/Admin/FOM-Anwendungsprojekt/Models/train-csv-r_t_e-r_i_n-m_d-d-b-AS_N-c_e-15kepochs_001.h5")
 
# Die Vorhersagen sollten nun nahe an den Zielvariablen liegen, da es sich um ein einfaches Beispiel handelt.


#testpath = "/Users/MK/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv"
#testdata = pd.read_csv(testpath, sep=';', decimal='.')
#x_t = testdata[['cleanliness_rating', 'guest_satisfaction_overall', 'AttractionScore_Norm']].values
#y_t = testdata[['realSum']].values
#predictions = model.predict(x_t)
#model.save("/mnt/c/Users/Admin/FOM-Anwendungsprojekt/Models/c_rt-g_s_o-as_n-100epochs_002.h5")