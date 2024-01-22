# Einteilung der Zufriedenheitsbewertungen in Kategorien
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from scipy import stats
import numpy as np

import contextily as ctx

df = pd.read_csv("/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Datenbeschreibung/Airbnb_Prices_V1.0_VorBereinigung.csv", sep=';',decimal='.', engine='python')



def plot_attraction_score_on_map(df, city_name):
    # Filtere die Daten für die angegebene Stadt
    city_data = df[df['city'] == city_name]
    gdf = gpd.GeoDataFrame(city_data, geometry=gpd.points_from_xy(city_data.lng, city_data.lat))
    
    # Setze das CRS für das GeoDataFrame auf WGS84 (epsg:4326)
    gdf.set_crs(epsg=4326, inplace=True)
    
    # Konvertiere das CRS in Web Mercator (epsg:3857) für contextily
    gdf = gdf.to_crs(epsg=3857)
    
    # Plotte die Punkte auf der Karte
    ax = gdf.plot(figsize=(10, 10), alpha=0.5, column='AttractionScore', cmap='viridis', legend=True)
    
    # Füge eine Grundkarte hinzu
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Entferne die Achsen
    ax.set_axis_off()
    
    # Setze den Titel
    #ax.set_title(f'AttractionScore nach Bereinigung in {city_name}', fontsize=25)
    
    # Zeige den Plot an
    plt.tight_layout()
    plt.savefig(f"Figures/AttractionScore/vorBereinigung_{city_name}.svg", format='svg')
    plt.show()

# Erhalte die Liste der einzigartigen Städte
unique_cities = df['city'].unique()

# Plotte für jede Stadt
for city in tqdm(unique_cities):
    plot_attraction_score_on_map(df, city)
