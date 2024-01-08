# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:02:26 2024

@author: Dennis
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 10:48:27 2024

@author: Dennis
"""

# Full code to create individual geospatial plots for each city in the dataset
import os
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd


airbnb_data = pd.read_csv("/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv", sep=';',decimal='.', engine='python')
# Konvertierung der Längen- und Breitengrade in ein Geopandas GeoDataFrame
geometry = [Point(xy) for xy in zip(airbnb_data['lng'], airbnb_data['lat'])]
geo_df = gpd.GeoDataFrame(airbnb_data, crs='EPSG:4326', geometry=geometry)

# Umwandlung in Web Mercator Projektion für die Kontextdarstellung
geo_df = geo_df.to_crs(epsg=3857)

# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Prepare a directory for elbow plots
os.makedirs('elbow_plots', exist_ok=True)

# Prepare a directory for clustered city plots
os.makedirs('clustered_city_plots', exist_ok=True)

# Function to perform Elbow method and KMeans clustering
def cluster_city(city_geo_df, city_name):
    # Extract coordinates
    coords = city_geo_df[['lat', 'lng']].to_numpy()
    
    # Standardize features
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Determine the optimal number of clusters using the Elbow method
    sum_of_squared_distances = []
    K = range(1, 10)
    for k in tqdm(K, desc=f'Elbow Method for {city_name}'):
        km = KMeans(n_clusters=k)
        km = km.fit(coords_scaled)
        sum_of_squared_distances.append(km.inertia_)
    
    # Plot the Elbow
    plt.figure()
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method For Optimal k for ' + city_name)
    plt.savefig(f'elbow_plots/{city_name}_elbow.png')
    plt.close()
    
    # Use silhouette score to find the optimal number of clusters
    silhouette_scores = [silhouette_score(coords_scaled, KMeans(n_clusters=k).fit(coords_scaled).labels_)
                        for k in range(2, 10)]
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    
    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k)
    city_geo_df['cluster'] = kmeans.fit_predict(coords_scaled)
    
    # Plot the clustered city
    ax = city_geo_df.plot(column='cluster', figsize=(10, 10), categorical=True, legend=True, alpha=0.5, edgecolor='k')
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    plt.savefig(f'clustered_city_plots/{city_name}_clustered.png')
    plt.close()
    
    return optimal_k

# Apply clustering for each city
optimal_clusters = {}
for city in tqdm(geo_df['city'].unique(), desc='Clustering Cities'):
    city_geo_df = geo_df[geo_df['city'] == city]
    optimal_clusters[city] = cluster_city(city_geo_df, city)

print('Clustering and plotting completed for all cities.')
print('Optimal number of clusters for each city:')
print(optimal_clusters)