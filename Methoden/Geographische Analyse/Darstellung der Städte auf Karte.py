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

# Create a directory for city plots if it doesn't exist
os.makedirs('city_plots', exist_ok=True)

# Generate a plot for each city
for city in tqdm(geo_df['city'].unique()):
    # Filter the data for the current city
    city_geo_df = geo_df[geo_df['city'] == city]
    
    # Plotting the data
    ax = city_geo_df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
    
    # Add the OpenStreetMap basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    
    # Save the plot to a PNG file
    plot_filename = f'city_plots/{city}_map_plot.png'
    ax.figure.savefig(plot_filename)
    plt.close(ax.figure)

print('All city plots have been saved in the city_plots directory.')
    
