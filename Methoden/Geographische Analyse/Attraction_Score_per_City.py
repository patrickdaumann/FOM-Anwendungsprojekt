# Einteilung der Zufriedenheitsbewertungen in Kategorien
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from scipy import stats
import numpy as np

import contextily as ctx

df = pd.read_csv("/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/AttractionScores.csv", sep=';',decimal='.', engine='python')




def plot_attraction_score_on_map(df, city_name):
    city_data = df[df['city'] == city_name]
    gdf = gpd.GeoDataFrame(city_data, geometry=gpd.points_from_xy(city_data.lng, city_data.lat))
    
    # Set the CRS for the GeoDataFrame to WGS84 (epsg:4326)
    gdf.set_crs(epsg=4326, inplace=True)
    
    # Convert the CRS to Web Mercator (epsg:3857) for contextily
    gdf = gdf.to_crs(epsg=3857)
    
    # Plot the points on the map
    ax = gdf.plot(figsize=(10, 10), alpha=0.5, column='AttractionScore_Norm', cmap='viridis', legend=True)
    
    # Add a basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Remove axis
    ax.set_axis_off()
    
    # Set title
    ax.set_title(f'Attraction Scores in {city_name}', fontsize=15)
    
    # Show the plot
    plt.show()

# Get the list of unique cities
unique_cities = df['city'].unique()

# Plot for each city
for city in tqdm(unique_cities):
    plot_attraction_score_on_map(df, city)