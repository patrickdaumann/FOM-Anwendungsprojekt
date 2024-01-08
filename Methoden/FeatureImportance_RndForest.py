# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 10:53:28 2024

@author: Dennis
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
file_path = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv"
data = pd.read_csv(file_path, sep=';', decimal='.')

data = data.drop("room_type", axis=1)



# Wählen Sie Ihre Features und die Zielvariable aus
#X = data.drop("realSum", axis=1)  # Beispielmerkmale

X = data[['cleanliness_rating', 'host_is_superhost', 'city','rest_index','attr_index' ]]
y = data['guest_satisfaction_overall']  # Zielvariable

X = pd.get_dummies(X, columns=['city'])

# Aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



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
plt.yticks(range(len(X.columns)), X.columns)
plt.xlabel('Feature Importance')
plt.show()
