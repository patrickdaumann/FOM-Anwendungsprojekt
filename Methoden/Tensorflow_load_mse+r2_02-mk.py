# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:49:24 2024

@author: kesper
"""
# Import der notwendigen Module
from tensorflow import keras
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.utils import plot_model
import csv


modelsroot = "/mnt/c/Users/MK/FOM-Anwendungsprojekt/Models/"

modelnames = ["train-csv-r_t_e-r_i_n-m_d-d-b-AS_N-c_e-5kepochs_002-PD.h5", 
              "train-csv-r_t_e-r_i_n-m_d-d-b-AS_N-c_e-10kepochs_002_PD.h5",
              "train-csv-r_t_e-r_i_n-m_d-d-b-AS_N-c_e-10kepochs_001.h5",
              "train-csv-r_t_e-r_i_n-m_d-d-b-AS_N-c_e-15kepochs_001.h5",
              "train-csv-r_t_e-r_i_n-m_d-d-b-AS_N-c_e-15kepochs_002-moooooreLayers.h5",
              "train-csv-r_t_e-r_i_n-m_d-d-b-AS_N-c_e-50epochs_001.h5"]

modeldata = []

for modelname in modelnames:
    # Daten vorbereiten
    df = pd.read_csv('/mnt/c/Users/MK/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv', sep=';', decimal='.')
    x = df[['room_type_encoded', 'rest_index_norm', 'metro_dist', 'dist', 'bedrooms', 'AttractionScore_Norm', 'city_encoded']].values
    y = df[['realSum_Normalized']].values
    maxrealSum = df[['realSum']].values.max()


    # Pfad zur H5-Datei
    model_path = f"{modelsroot}{modelname}"

    # Laden des Modells
    model = keras.models.load_model(model_path)

    # Modellübersicht anzeigen
    model.summary()

    # Vorhersagen treffen
    predictions = model.predict(x)

    # Berechnung von Mean Squared Error und R² Score
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)


    modelinfo = {
        "modelname": modelname,
        "R2": r2,
        "MSE": mse 
    }

    modeldata.append(modelinfo)

    # Ergebnisse anzeigen
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    plot_model(model, to_file='/mnt/c/Users/MK/FOM-Anwendungsprojekt/Figures/Models/model_03.png', show_shapes=True, show_layer_names=True)

print(modeldata)

headers = ["modelname", "R2", "MSE"]

#with open("C:\\Users\\patrick\\GitHub\\FOM-Anwendungsprojekt\\Data\\Output\\modelData.csv", "w") as file:
#    writer = csv.DictWriter(file, fieldnames=headers)
#    writer.writeheader()
#    writer.writerows(modeldata)