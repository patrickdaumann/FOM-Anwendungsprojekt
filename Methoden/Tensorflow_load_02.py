# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:57:44 2024

@author: kesper
"""

from tensorflow import keras
import pandas as pd

df = pd.read_csv('/mnt/c/Users/MK/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv', sep=';', decimal='.')
x = df[['attr_index_norm', 'rest_index_norm', 'person_capacity', 'AttractionScore_Norm']].values
#x = x.astype('float32')  # Konvertieren in float32

# Pfad zur H5-Datei
model_path = '/mnt/c/Users/MK/FOM-Anwendungsprojekt/Models/train-csv-a_i_n-r_i_n-p_c-AS_N-1000epochs_001.h5'

# Laden des Modells
model = keras.models.load_model(model_path)

# Modellübersicht anzeigen
model.summary()

# Angenommen, `some_input_data` ist Ihre Eingabedaten
predictions = model.predict(x)

# Ergebnisse anzeigen oder weiterverarbeiten
print(predictions)

#test