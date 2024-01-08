# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 11:57:43 2024

@author: Dennis
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the CSV file into a DataFrame
file_path = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv"
data = pd.read_csv(file_path, sep=';', decimal='.')

data = data.drop("room_type", axis=1)


# Umwandeln kategorialer Daten in numerische (falls vorhanden)
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Auswahl der Features für die PCA
X = data.drop('guest_satisfaction_overall', axis=1)  # Ersetzen Sie 'Zielvariable' mit dem Namen Ihrer Zielvariablen

# Standardisieren der Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.decomposition import PCA

# PCA mit einer bestimmten Anzahl von Komponenten
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X_scaled)

# Erstellen eines DataFrame für die Hauptkomponenten
pca_df = pd.DataFrame(data=principalComponents, columns=['Hauptkomponente 1', 'Hauptkomponente 2','Hauptkomponente 3'])


import matplotlib.pyplot as plt



print("Erklärte Varianz pro Hauptkomponente:", pca.explained_variance_ratio_)


plt.figure(figsize=(8,6))
plt.plot(pca.explained_variance_ratio_, 'o-')
plt.xlabel('Hauptkomponente')
plt.ylabel('Varianzanteil der einzelnen Hauptkomponenten')
plt.title('Scree-Plot')
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# PCA mit 3 Hauptkomponenten
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X_scaled)

# Erstellen eines DataFrame für die Hauptkomponenten
pca_df = pd.DataFrame(data=principalComponents, columns=['Hauptkomponente 1', 'Hauptkomponente 2', 'Hauptkomponente 3'])

# 3D-Scatterplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pca_df['Hauptkomponente 1'], pca_df['Hauptkomponente 2'], pca_df['Hauptkomponente 3'])

ax.set_xlabel('Hauptkomponente 1')
ax.set_ylabel('Hauptkomponente 2')
ax.set_zlabel('Hauptkomponente 3')
plt.title('3D-Darstellung der ersten drei Hauptkomponenten')
plt.show()



# 2D-Scatterplots für jede Kombination von Hauptkomponenten
plt.figure(figsize=(15, 5))

# Plot 1: Hauptkomponente 1 und 2
plt.subplot(1, 3, 1)
plt.scatter(pca_df['Hauptkomponente 1'], pca_df['Hauptkomponente 2'])
plt.xlabel('Hauptkomponente 1')
plt.ylabel('Hauptkomponente 2')
plt.title('Hauptkomponente 1 vs 2')

# Plot 2: Hauptkomponente 1 und 3
plt.subplot(1, 3, 2)
plt.scatter(pca_df['Hauptkomponente 1'], pca_df['Hauptkomponente 3'])
plt.xlabel('Hauptkomponente 1')
plt.ylabel('Hauptkomponente 3')
plt.title('Hauptkomponente 1 vs 3')

# Plot 3: Hauptkomponente 2 und 3
plt.subplot(1, 3, 3)
plt.scatter(pca_df['Hauptkomponente 2'], pca_df['Hauptkomponente 3'])
plt.xlabel('Hauptkomponente 2')
plt.ylabel('Hauptkomponente 3')
plt.title('Hauptkomponente 2 vs 3')

plt.tight_layout()
plt.show()