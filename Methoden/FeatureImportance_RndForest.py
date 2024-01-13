# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 10:53:28 2024

@author: Dennis
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Annahme: Ihre Daten sind in einer Datei mit dem Namen "data.csv"
path = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Train.csv"

path2 = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv"

data = pd.read_csv(path, sep=';', decimal='.')
data_test = pd.read_csv(path2, sep=';', decimal='.')


# Wählen Sie Ihre Features und die Zielvariable aus
#X = data.drop("realSum", axis=1)  # Beispielmerkmale

# Definieren Sie erneut abhängige und unabhängige Variablen
X_train = data[['AttractionScore_Norm', "dist", "metro_dist", "guest_satisfaction_overall", "cleanliness_rating", "rest_index_norm", "host_is_superhost", "city_encoded","bedrooms","room_type_encoded" ]]
y_train = data["realSum_Normalized"]


X_test = data_test[['AttractionScore_Norm', "dist", "metro_dist", "guest_satisfaction_overall", "cleanliness_rating", "rest_index_norm", "host_is_superhost", "city_encoded","bedrooms","room_type_encoded" ]]
y_test = data_test["realSum_Normalized"]


from sklearn.ensemble import RandomForestRegressor

# Random Forest Regressionsmodell erstellen
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Modell mit Trainingsdaten trainieren
rf_regressor.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, r2_score

# Vorhersagen mit dem Testset machen
y_pred = rf_regressor.predict(X_test)

# Ergebnisse ausgeben
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

import matplotlib.pyplot as plt

# Feature Importance
feature_importances = rf_regressor.feature_importances_

# Visualisierung
plt.barh(range(len(feature_importances)), feature_importances)
plt.yticks(range(len(X_train.columns)), X_train.columns)
plt.xlabel('Feature Importance')
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
