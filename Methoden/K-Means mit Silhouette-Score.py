# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 10:05:40 2023

@author: Dennis
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler as ss
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


path = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv"

data = pd.read_csv(path, sep=';', decimal='.')

# Wählen Sie die relevanten Merkmale für das Clustering aus
selected_features = data[["realSum_Normalized", "cleanliness_rating", "room_type_encoded","guest_satisfaction_overall","bedrooms","dist", "metro_dist", "attr_index_norm", "rest_index_norm", "lng", "lat"]]

# Überprüfen Sie, ob die ausgewählten Merkmale korrekt sind
print(selected_features.head())

# Standardisieren Sie die ausgewählten Merkmale
scaler = ss()
scaled_features = scaler.fit_transform(selected_features)

# Überprüfen Sie die standardisierten Merkmale
print(scaled_features[:5])




# Liste von Werten für K (Anzahl der Cluster)#


k_values = range(2, 12)





#Berechnen Sie die Inertia (innerhalb der Cluster Sum of Squares) für verschiedene Werte von K
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(8, 4))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Silhouette-Analyse zur Bestimmung von K')
plt.xlabel('Anzahl der Cluster (K)')
plt.ylabel('Silhouette-Score')
plt.grid()
plt.show(



#  Wählen Sie die optimale Anzahl von Clustern basierend auf der Elbow-Methode (z.B., K=4)
# optimal_k = 4
# kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# cluster_labels = kmeans.fit_predict(scaled_features)

#  Fügen Sie die Clusterzuordnungen dem ursprünglichen DataFrame hinzu
# data['Cluster'] = cluster_labels

#  Überprüfen Sie die Zuordnungen
# print(data.head())




# plt.scatter(data['Price'], data['Person Capacity'], c=data['Cluster'], cmap='viridis')
# plt.xlabel('Price')
# plt.ylabel('Person Capacity')
# plt.title('2D Scatterplot mit Cluster-Färbung')
# plt.show()

# plt.scatter(data['Price'], data['Cleanliness Rating'], c=data['Cluster'], cmap='viridis')
# plt.xlabel('Price')
# plt.ylabel('Cleanliness Rating')
# plt.title('2D Scatterplot mit Cluster-Färbung')
# plt.show()

# plt.scatter(data['Price'], data['Guest Satisfaction'], c=data['Cluster'], cmap='viridis')
# plt.xlabel('Price')
# plt.ylabel('Guest Satisfaction')
# plt.title('2D Scatterplot mit Cluster-Färbung')
# plt.show()

# plt.scatter(data['Price'], data['City Center (km)'], c=data['Cluster'], cmap='viridis')
# plt.xlabel('Price')
# plt.ylabel('City Center (km)')
# plt.title('2D Scatterplot mit Cluster-Färbung')
# plt.show()

# plt.scatter(data['Price'], data['Metro Distance (km)'], c=data['Cluster'], cmap='viridis')
# plt.xlabel('Price')
# plt.ylabel('Metro Distance (km)')
# plt.title('2D Scatterplot mit Cluster-Färbung')
# plt.show()

# plt.scatter(data['Price'], data['Attraction Index'], c=data['Cluster'], cmap='viridis')
# plt.xlabel('Price')
# plt.ylabel('Attraction Index')
# plt.title('2D Scatterplot mit Cluster-Färbung')
# plt.show()

# plt.scatter(data['Price'], data['Restraunt Index'], c=data['Cluster'], cmap='viridis')
# plt.xlabel('Price')
# plt.ylabel('Restraunt Index')
# plt.title('2D Scatterplot mit Cluster-Färbung')
# plt.show()
