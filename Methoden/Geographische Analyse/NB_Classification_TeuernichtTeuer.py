# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:45:16 2024

@author: Dennis
"""
import pandas as pd
# Annahme: Ihre Daten sind in einer Datei mit dem Namen "data.csv"
path = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/AttractionScoresRealSum_cleaned.csv"

data = pd.read_csv(path, sep=';', decimal='.')

# Bestimmung des Schwellenwerts f�r die Klassifizierung
# Wir verwenden das obere Quartil (75%) als Schwellenwert f�r 'teuer'
threshold = data['realSum_Normalized'].quantile(0.75)
print('Schwellenwert f�r teuer:', threshold)

# Hinzuf�gen einer neuen Spalte f�r die Klassifizierung
# Wenn der Preis �ber dem Schwellenwert liegt, ist das Appartement 'teuer' (1), sonst 'nicht teuer' (0)
data['is_expensive'] = (data['realSum_Normalized'] > threshold).astype(int)

# Anzeigen der aktualisierten Tabelle mit der neuen Spalte
print(data[['realSum_Normalized', 'is_expensive']].head())

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from shapely.geometry import Point
import seaborn as sns

# Auswahl der Features und des Targets
X = data[['room_type_encoded', 'city_encoded', 'daytype_encoded', 'AttractionScore_Norm']]
y = data['is_expensive']

# Aufteilen der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Erstellen und Trainieren des NaiveBayes Klassifikators
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Vorhersagen auf dem Testset
y_pred = nb_classifier.predict(X_test)

# Erstellen der Konfusionsmatrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Konfusionsmatrix:\n', conf_matrix)

accuracy = accuracy_score(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Vorhergesagtes Label')
plt.ylabel('Echtes Label')
plt.show()
print('Accuracy of the classifier: {:.2f}%'.format(accuracy * 100))
# Adjusting the code to add explicit labels for expensive and not expensive points on the plot.
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Assuming 'data' is the DataFrame with the necessary columns 'lat', 'lng', 'city', and 'is_expensive'
# Convert the DataFrame to a GeoDataFrame
# First, create a 'geometry' column with point data
data['geometry'] = data.apply(lambda row: Point(row['lng'], row['lat']), axis=1)

# Now convert the DataFrame to a GeoDataFrame
# Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')

# Reproject the data to Web Mercator (EPSG:3857) for use with contextily
gdf = gdf.to_crs(epsg=3857)

# Create a plot for each city
for city in gdf['city'].unique():
    # Filter data for the current city
    city_data = gdf[gdf['city'] == city]
    
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot expensive points
    not_expensive = city_data[city_data['is_expensive'] == False]
    ax.scatter(not_expensive.geometry.x, not_expensive.geometry.y, color='green', label='Nicht Teuer')
    expensive = city_data[city_data['is_expensive'] == True]
    ax.scatter(expensive.geometry.x, expensive.geometry.y, color='red', label='Teuer')
    # Plot not expensive points

    # Add the base map
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    # Remove axis
    ax.set_axis_off()
    # Add legend
    ax.legend()
    # Save the plot to a file
    
    ax.set_title(f'Klassifikation von {city}', fontsize=15)
    
  
    plt.show

