# -*- coding: utf-8 -*-
"""
Erstellt am Sonntag, 7. Januar 2024

Autor: Dennis
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Daten laden
df = pd.read_csv("/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Full.csv", sep=';', decimal='.', engine='python')
# Da die kategorischen Variablen bereits codiert sind, können wir mit der Analyse fortfahren

# Aufteilen der Daten
X = df.drop(['guest_satisfaction_overall', 'room_type', 'city', 'daytype','lat','lng','room_shared','room_private', 'AttractionScore','realSum','realSum_Normalized','rest_index','attr_index','attr_index_norm'], axis=1)  # Zielvariable und normalisierte Version ausschließen
y = df['realSum_Normalized']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell trainieren
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Vorhersagen und Auswertung
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Feature Importance (Bedeutung der Merkmale)
feature_importance = rf.feature_importances_

# Vorbereiten der Feature Importance für die Visualisierung
features = X_train.columns
importances = rf.feature_importances_
indices = np.argsort(importances)

# Visualisierung der Feature Importance
plt.figure(figsize=(10, 6))
plt.title('Bedeutung der Merkmale')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Bedeutung')
plt.tight_layout()
plt.savefig(f"Figures/RandomForest/FeatureImportance.svg", format='svg')
plt.show()

# Ausgabe des MSE (Mittlerer quadratischer Fehler)
print('Mittlerer quadratischer Fehler (MSE):', mse)
