import os
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Laden der Daten
airbnb_data = pd.read_csv("/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv", sep=';', decimal='.', engine='python')

# Konvertierung in ein GeoDataFrame
geometry = [Point(xy) for xy in zip(airbnb_data['lng'], airbnb_data['lat'])]
geo_df = gpd.GeoDataFrame(airbnb_data, crs='EPSG:4326', geometry=geometry)

# Umwandlung in Web Mercator Projektion
geo_df = geo_df.to_crs(epsg=3857)

# Funktion zur Durchführung der Clusterbildung
def cluster_city(city_geo_df, city_name):
    # Daten für die Clusterbildung vorbereiten
    real_sum_values = city_geo_df[['realSum']].to_numpy()

    # Standardisierung der Daten
    scaler = StandardScaler()
    real_sum_scaled = scaler.fit_transform(real_sum_values)

    # Elbow-Methode zur Bestimmung der optimalen Clusteranzahl
    sum_of_squared_distances = []
    K = range(1, 10)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(real_sum_scaled)
        sum_of_squared_distances.append(km.inertia_)

    # Elbow-Plot für jede Stadt
    plt.figure()
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Summe der quadratischen Abstände')
    plt.title(f'Elbow-Methode für optimales k für {city_name}')
    plt.savefig(f'elbow_plots_realSum/{city_name}_elbow.png')
    plt.close()

    # Bestimmung der optimalen Clusteranzahl
    silhouette_scores = [silhouette_score(real_sum_scaled, KMeans(n_clusters=k).fit(real_sum_scaled).labels_)
                         for k in range(2, 10)]
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2

    # Durchführung der KMeans-Clusterbildung
    kmeans = KMeans(n_clusters=6)
    city_geo_df['cluster'] = kmeans.fit_predict(real_sum_scaled)

    # Visualisierung der Cluster auf der Karte
    ax = city_geo_df.plot(column='cluster', figsize=(10, 10), categorical=True, legend=True, alpha=0.5, edgecolor='k')
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    plt.savefig(f'clustered_city_plots_realSum/{city_name}_clustered.png')
    plt.close()

    return optimal_k

# Anwendung der Clusterbildung für jede Stadt
optimal_clusters = {}
for city in tqdm(geo_df['city'].unique(), desc='Clustering Cities'):
    city_geo_df = geo_df[geo_df['city'] == city]
    optimal_clusters[city] = cluster_city(city_geo_df, city)

print('Clustering and plotting completed for all cities.')
print('Optimal number of clusters for each city:')
print(optimal_clusters)
