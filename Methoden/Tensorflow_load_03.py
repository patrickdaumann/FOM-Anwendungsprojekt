# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:31:09 2024

@author: kesper
"""

from tensorflow import keras
import pandas as pd

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

# Vorhersagen treffen
predictions = model.predict(x)

# DataFrame für Ergebnisse
results_df = pd.DataFrame(columns=['Prediction', 'Actual', 'Delta'])

# Vorhersagen und Deltas sammeln
for i in range(len(test)):
    delta = abs(predictions[i] - test[i]) * maxrealSum
    results_df = results_df.append({'Prediction': predictions[i][0], 'Actual': test[i][0], 'Delta': delta[0]}, ignore_index=True)

# Ergebnisse in CSV speichern
results_df.to_csv('/mnt/c/Users/Admin/FOM-Anwendungsprojekt/Results/predictions_01.csv', index=False)

# Durchschnittliches Delta berechnen
average_delta = results_df['Delta'].mean()
print("Durchschnittliches Delta:", average_delta)