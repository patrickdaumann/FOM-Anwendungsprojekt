# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 13:46:09 2024

@author: Dennis
"""


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv", sep=';', decimal='.', engine='python')
# Since the categorical variables are already encoded, we can proceed with the analysis

# Splitting the data
X = df.drop(['guest_satisfaction_overall', 'room_type', 'city', 'daytype','lat','lng'], axis=1)  # Drop target variable and normalized version
y = df['guest_satisfaction_overall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predicting and evaluating
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Feature importance
feature_importance = rf.feature_importances_

# Preparing feature importance for visualization
features = X_train.columns
importances = rf.feature_importances_
indices = np.argsort(importances)

# Visualizing feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Output the MSE
print('Mean Squared Error:', mse)