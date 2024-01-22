# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:57:44 2024

@author: kesper
"""
# Import der notwendigen Module
from tensorflow import keras
import pandas as pd

# Daten vorbereiten
df = pd.read_csv('/mnt/c/Users/Admin/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv', sep=';', decimal='.')
x = df[['room_type_encoded', 'rest_index_norm', 'metro_dist', 'dist', 'bedrooms', 'AttractionScore_Norm', 'city_encoded']].values
test = df[['realSum_Normalized']].values
maxrealSum = df[['realSum']].values.max()
#x = x.astype('float32')  # Konvertieren in float32

# Pfad zur H5-Datei
model_path = '/mnt/c/Users/Admin/FOM-Anwendungsprojekt/Models/train-csv-r_t_e-r_i_n-m_d-d-b-AS_N-c_e-10kepochs_001.h5'

# Laden des Modells
model = keras.models.load_model(model_path)

# Modellübersicht anzeigen
model.summary()

# x = Eingabedaten
predictions = model.predict(x)

# Ergebnisse anzeigen oder weiterverarbeiten
print(predictions)

for i in range(0,len(test)):
    print(f"{predictions[i]};{test[i]};Delta:{abs(predictions[i]-test[i])*maxrealSum}$")
