# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 09:31:59 2024

@author: Dennis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
from IPython.display import display

# Einlesen der CSV-Datei
file_path = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv"
airbnb_data = pd.read_csv(file_path, sep=';', decimal='.')

# Gruppierung der Daten nach Stadt
grouped_data = airbnb_data.groupby('city').agg({'realSum': ['mean', 'median', 'std'], 'guest_satisfaction_overall': ['mean', 'median', 'std']})

# Visualisierung der Preisverteilung nach Stadt
plt.figure(figsize=(15, 6))
sns.barplot(x=grouped_data.index, y=grouped_data['realSum']['mean'])
plt.xticks(rotation=45)
plt.title('Durchschnittspreis nach Stadt')
plt.ylabel('Durchschnittspreis')
plt.xlabel('Stadt')

# Visualisierung der Bewertungsverteilung nach Stadt
plt.figure(figsize=(15, 6))
sns.barplot(x=grouped_data.index, y=grouped_data['guest_satisfaction_overall']['mean'])
plt.xticks(rotation=45)
plt.title('Durchschnittliche Gästebewertung nach Stadt')
plt.ylabel('Durchschnittliche Bewertung')
plt.xlabel('Stadt')

plt.show()



#Erstellen der Karten



# Angenommen, Ihr DataFrame heißt airbnb_data und hat Spalten 'city', 'lat', 'lng'
for stadt in airbnb_data['city'].unique():
    # Filtern der Daten für die aktuelle Stadt
    stadt_data = airbnb_data[airbnb_data['city'] == stadt]

    # Berechnen der durchschnittlichen Koordinaten für die Stadt
    durchschnitt_lat = stadt_data['lat'].mean()
    durchschnitt_lng = stadt_data['lng'].mean()

    # Erstellen einer Karte für die Stadt
    stadt_map = folium.Map(location=[durchschnitt_lat, durchschnitt_lng], zoom_start=12)

    # Hinzufügen von Markierungen für jede Unterkunft in der Stadt
    for idx, row in stadt_data.iterrows():
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=f"Preis: {row['realSum']}€ - Bewertung: {row['guest_satisfaction_overall']}/100"
        ).add_to(stadt_map)

    # Speichern der Karte als HTML
    stadt_map.save(f'{stadt}_map.html')
