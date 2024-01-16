# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
 
# Beispiel-Daten: Ein eindimensionales Array von Features und zugehörige Zielvariablen
#X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
#y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

# Annahme: Ihre Daten sind in einer Datei mit dem Namen "data.csv"
path = "/Users/MK/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv"

data = pd.read_csv(path, sep=';', decimal='.')

x = data[['cleanliness_rating', 'guests_satisfaction_overall', 'attractionscore_norm']]
y = data[['realSum']]
 
# Erstellen eines einfachen neuronalen Netzwerks
model = keras.Sequential([
    layers.Dense(1, input_shape=(3,))  # Ein einzelnes Neuron, da wir nur eine Dimension haben
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
 
# Die Vorhersagen sollten nun nahe an den Zielvariablen liegen, da es sich um ein einfaches Beispiel handelt.