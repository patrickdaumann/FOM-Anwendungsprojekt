# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:19:44 2024

@author: Dennis
"""

# Durchführung einer linearen Regression
import pandas as pd


data = pd.read_csv("/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv", sep=';',decimal='.', engine='python')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Feature Engineering und Auswahl relevanter Merkmale
features = data[['person_capacity', 'bedrooms', 'cleanliness_rating', 'dist', 'metro_dist', 'attr_index', 'rest_index', 'realSum']]

# Normalisierung der Merkmale
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Definition der Zielvariable
target = data['guest_satisfaction_overall']

# Erneutes Trainieren des Modells
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Bewertung des erweiterten Modells
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Visualisierung der tatsächlichen vs. vorhergesagten Preise
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=predictions)
plt.xlabel('Tatsächliche Preise')
plt.ylabel('Vorhergesagte Preise')
plt.title('Tatsächliche vs. Vorhergesagte Preise')
plt.show()