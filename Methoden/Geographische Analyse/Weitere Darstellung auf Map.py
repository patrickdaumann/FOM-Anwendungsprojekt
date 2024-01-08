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