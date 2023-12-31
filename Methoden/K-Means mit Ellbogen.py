# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler as ss
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "C:/Users/Dennis/Documents/Aemf1_cleaned_5000.csv"

data = pd.read_csv(path, sep=';', decimal=',')

# Wählen Sie die relevanten Merkmale für das Clustering aus
selected_features = data[["Price", "Person Capacity", "Cleanliness Rating", "Guest Satisfaction", "City Center (km)", "Metro Distance (km)", "Normalised Attraction Index", "Normalised Restraunt Index", "lng", "lat"]]

# Überprüfen Sie, ob die ausgewählten Merkmale korrekt sind
print(selected_features.head())

# Standardisieren Sie die ausgewählten Merkmale
scaler = ss()
scaled_features = scaler.fit_transform(selected_features)

# Überprüfen Sie die standardisierten Merkmale
print(scaled_features[:5])




# Liste von Werten für K (Anzahl der Cluster)#


k_values = range(2, 11)
inertia_values = []




#Berechnen Sie die Inertia (innerhalb der Cluster Sum of Squares) für verschiedene Werte von K
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia_values.append(kmeans.inertia_)

# Plot der Elbow-Methode
plt.figure(figsize=(8, 4))
plt.plot(k_values, inertia_values, marker='o', linestyle='-', color='b')
plt.title('Elbow-Methode zur Bestimmung von K')
plt.xlabel('Anzahl der Cluster (K)')
plt.ylabel('Inertia')
plt.grid()
plt.show()



# Wählen Sie die optimale Anzahl von Clustern basierend auf der Elbow-Methode (z.B., K=4)
optimal_k = 7
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# Fügen Sie die Clusterzuordnungen dem ursprünglichen DataFrame hinzu
data['Cluster'] = cluster_labels

# Überprüfen Sie die Zuordnungen
print(data.head())




plt.scatter(data['Price'], data['Person Capacity'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Price')
plt.ylabel('Person Capacity')
plt.title('2D Scatterplot mit Cluster-Färbung')
plt.show()

plt.scatter(data['Price'], data['Cleanliness Rating'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Price')
plt.ylabel('Cleanliness Rating')
plt.title('2D Scatterplot mit Cluster-Färbung')
plt.show()

plt.scatter(data['Price'], data['Guest Satisfaction'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Price')
plt.ylabel('Guest Satisfaction')
plt.title('2D Scatterplot mit Cluster-Färbung')
plt.show()

plt.scatter(data['Price'], data['City Center (km)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Price')
plt.ylabel('City Center (km)')
plt.title('2D Scatterplot mit Cluster-Färbung')
plt.show()

plt.scatter(data['Price'], data['Metro Distance (km)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Price')
plt.ylabel('Metro Distance (km)')
plt.title('2D Scatterplot mit Cluster-Färbung')
plt.show()

plt.scatter(data['Price'], data['Normalised Attraction Index'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Price')
plt.ylabel('Normalised Attraction Index')
plt.title('2D Scatterplot mit Cluster-Färbung')
plt.show()

plt.scatter(data['Price'], data['Normalised Restraunt Index'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Price')
plt.ylabel('Normalised Restraunt Index')
plt.title('2D Scatterplot mit Cluster-Färbung')
plt.show()

plt.scatter(data['Price'], data['lng'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Price')
plt.ylabel('lng')
plt.title('2D Scatterplot mit Cluster-Färbung')
plt.show()

plt.scatter(data['Price'], data['lat'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Price')
plt.ylabel('lat')
plt.title('2D Scatterplot mit Cluster-Färbung')
plt.show()


fig = plt.figure(figsize=(10, 100))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Price'], data['City Center (km)'], data['Cleanliness Rating'], c=data['Cluster'], cmap='viridis')
ax.set_xlabel('Price')
ax.set_ylabel('City Center (km)')
ax.set_zlabel('Cleanliness Rating')
ax.zaxis.labelpad = -2


# ax.set_xlim(0,1000)
# ax.set_ylim(5,20)
# ax.set_zlim(2,8)



plt.title(f'3D Scatterplot mit Cluster-Färbung von {optimal_k} Clustern')
plt.show()



cluster_centers = kmeans.cluster_centers_  # Annahme: Sie haben bereits Ihren K-Means-Algorithmus durchgeführt
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', s=100, c='red', label='Cluster Centers')
plt.scatter(data['Price'], data['City Center (km)'], c=data['Cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('Price')
plt.ylabel('City Center (km)')
plt.title('Cluster-Zentren und 2D Scatterplot')
plt.legend()
plt.show()
