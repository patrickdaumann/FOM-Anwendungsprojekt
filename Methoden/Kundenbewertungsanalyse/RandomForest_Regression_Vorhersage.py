# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:21:23 2024

@author: Dennis
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Daten einlesen
path = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Train.csv"
path2 = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv"

data = pd.read_csv(path, sep=';', decimal='.')
data_test = pd.read_csv(path2, sep=';', decimal='.')

# Definieren der abhängigen und unabhängigen Variablen
X_train = data[[ "AttractionScore_Norm", "dist", "metro_dist",  "rest_index_norm", "city_encoded", "room_type_encoded", "bedrooms"]]
y_train = data["realSum_Normalized"]

X_test = data_test[[ "AttractionScore_Norm", "dist", "metro_dist",   "rest_index_norm", "city_encoded", "room_type_encoded", "bedrooms"]]
y_test = data_test["realSum_Normalized"]

# Random Forest Regressionsmodell
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Bewertung des Random Forest Modells
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f"Random Forest - Mean Squared Error: {rf_mse}")
print(f"Random Forest - R^2 Score: {rf_r2}")

# Visualisierung der tatsächlichen vs. vorhergesagten Preise
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=rf_predictions)
plt.xlabel('Tatsächliche Preise')
plt.ylabel('Vorhergesagte Preise (Random Forest)')
plt.title('Tatsächliche vs. Vorhergesagte Preise (Random Forest)')
plt.show()


# import numpy as np

# # Range der n_estimators, die getestet werden sollen
# n_estimators_range = np.arange(10, 200, 10)

# # Listen für die Speicherung der Scores
# r2_scores = []

# for n_estimators in n_estimators_range:
#     rf_regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
#     rf_regressor.fit(X_train, y_train)
#     y_pred = rf_regressor.predict(X_test)
#     r2 = r2_score(y_test, y_pred)
#     r2_scores.append(r2)

# # Visualisierung
# plt.figure(figsize=(10, 6))
# plt.plot(n_estimators_range, r2_scores, marker='o')
# plt.xlabel('Anzahl der Bäume (n_estimators)')
# plt.ylabel('R^2 Score')
# plt.title('R^2 Score in Abhängigkeit von der Anzahl der Bäume im Random Forest')
# plt.grid(True)
# plt.show()
