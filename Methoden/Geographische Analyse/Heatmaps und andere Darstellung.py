import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import matplotlib.pyplot as plt
import pandas as pd

airbnb_data = pd.read_csv("/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv", sep=';',decimal='.', engine='python')
# Konvertierung der Längen- und Breitengrade in ein Geopandas GeoDataFrame
geometry = [Point(xy) for xy in zip(airbnb_data['lng'], airbnb_data['lat'])]
geo_df = gpd.GeoDataFrame(airbnb_data, crs='EPSG:4326', geometry=geometry)

# Umwandlung in Web Mercator Projektion für die Kontextdarstellung
geo_df = geo_df.to_crs(epsg=3857)

# Erstellung der geographischen Plot
ax = geo_df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')

# Hinzufügen des Basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

# Einstellung der Achsen
ax.set_axis_off()

plt.show()


import math

# Berechnung der Anzahl der benötigten Subplots basierend auf der Anzahl der einzigartigen Städte
unique_cities = geo_df['city'].unique()
num_cities = len(unique_cities)
num_rows = math.ceil(num_cities / 2)

# Erstellung einer Heatmap für jede Stadt
fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, num_rows * 7.5))
axs = axs.flatten() # Umwandlung in ein 1D-Array für einfacheren Zugriff

for i, city in enumerate(unique_cities):
    # Filterung des DataFrames für jede Stadt
    city_df = geo_df[geo_df['city'] == city]
    
    # Erstellung der Heatmap
    if not city_df.empty:
        axs[i].set_title('Heatmap von ' + city)
        hb = axs[i].hexbin(city_df.geometry.x, city_df.geometry.y, gridsize=50, cmap='inferno', bins='log')
        fig.colorbar(hb, ax=axs[i])
        ctx.add_basemap(axs[i], source=ctx.providers.CartoDB.Positron)
        axs[i].set_axis_off()

# Anpassung des Layouts
plt.tight_layout()
plt.show()