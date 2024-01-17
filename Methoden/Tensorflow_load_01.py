# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 18:50:12 2024

@author: kesper
"""

from tensorflow import keras
import pandas as pd

df = pd.read_csv('C:/DATA/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Train.csv', sep=';', decimal='.')
x = df[['attr_index_norm', 'rest_index_norm', 'person_capacity', 'AttractionScore_Norm']].values
# Pfad zur H5-Datei
model_path = 'C:/DATA/Documents/GitHub/FOM-Anwendungsprojekt/Models/train-csv-a_i_n-r_i_n-p_c-AS_N-1000epochs_001.h5'

# Laden des Modells
model = keras.models.load_model(model_path)

# Modellübersicht anzeigen
model.summary()

# Angenommen, `some_input_data` ist Ihre Eingabedaten
predictions = model.predict(df)

# Ergebnisse anzeigen oder weiterverarbeiten
print(predictions)
