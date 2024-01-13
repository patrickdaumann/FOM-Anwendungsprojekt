# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:19:44 2024

@author: Dennis
"""

import pandas as pd
# Annahme: Ihre Daten sind in einer Datei mit dem Namen "data.csv"
path = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Train.csv"

path2 = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv"

data = pd.read_csv(path, sep=';', decimal='.')
data_test = pd.read_csv(path2, sep=';', decimal='.')

# Durchführung einer linearen Regression
# Definieren Sie erneut abhängige und unabhängige Variablen
X_train = data[['city_encoded', 'bedrooms', 'AttractionScore_Norm', 'room_type_encoded', 'rest_index_norm' ,'dist','metro_dist']]
y_train = data["realSum_Normalized"]


X_test = data_test[['city_encoded', 'bedrooms', 'AttractionScore_Norm', 'room_type_encoded', 'rest_index_norm' ,'dist','metro_dist']]
y_test = data_test["realSum_Normalized"]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns




model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Bewertung des erweiterten Modells
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Visualisierung der tatsächlichen vs. vorhergesagten Preise
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=predictions)
plt.xlabel('Tatsächliche Preise')
plt.ylabel('Vorhergesagte Preise')
plt.title('Tatsächliche vs. Vorhergesagte Preise')
plt.show()