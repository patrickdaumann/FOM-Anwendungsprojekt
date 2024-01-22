# -*- coding: utf-8 -*-
"""
Erstellt am Freitag, 12. Januar 2024

Autor: Dennis
"""
import pandas as pd
# Annahme: Ihre Daten sind in einer Datei mit dem Namen "data.csv"
pfad = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Train.csv"

pfad2 = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv"

daten = pd.read_csv(pfad, sep=';', decimal='.')
daten_test = pd.read_csv(pfad2, sep=';', decimal='.')



# Bestimmung des Schwellenwerts für die Klassifizierung
# Wir verwenden das obere Quartil (75%) als Schwellenwert für 'teuer'
schwellenwert = daten['realSum_Normalized'].quantile(0.75)
print('Schwellenwert für teuer:', schwellenwert)

# Hinzufügen einer neuen Spalte für die Klassifizierung
# Wenn der Preis über dem Schwellenwert liegt, ist die Wohnung 'teuer' (1), sonst 'nicht teuer' (0)
daten['is_expensive'] = (daten['realSum_Normalized'] > schwellenwert).astype(int)
daten_test['is_expensive'] = (daten['realSum_Normalized'] > schwellenwert).astype(int)

# Anzeigen der aktualisierten Tabelle mit der neuen Spalte
print(daten[['realSum_Normalized', 'is_expensive']].head())

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from shapely.geometry import Point
import seaborn as sns



# Wählen Sie Ihre Features und die Zielvariable aus
#X = daten.drop("realSum", axis=1)  # Beispielmerkmale

# Definieren Sie erneut abhängige und unabhängige Variablen
X_train = daten[['AttractionScore_Norm', "dist", "metro_dist",  "city_encoded"]]
y_train = daten["is_expensive"]


X_test = daten_test[[ "AttractionScore_Norm", "dist", "metro_dist", "city_encoded"]]
y_test = daten_test["is_expensive"]


# Erstellen und Trainieren des NaiveBayes Klassifikators
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Vorhersagen auf dem Testset
y_pred = nb_classifier.predict(X_test)

# Erstellen der Konfusionsmatrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Konfusionsmatrix:\n', conf_matrix)

genauigkeit = accuracy_score(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues')
#plt.title('Konfusionsmatrix')
plt.xlabel('Vorhergesagtes Label')
plt.ylabel('Echtes Label')
plt.tight_layout()
plt.savefig(f"Figures/Klassifikation/Matrix.svg", format='svg')
plt.show()
print('Genauigkeit des Klassifikators: {:.2f}%'.format(genauigkeit * 100))
# Anpassung des Codes, um explizite Labels für teure und nicht teure Punkte in der Grafik hinzuzufügen.
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Vorausgesetzt, 'daten' ist der DataFrame mit den erforderlichen Spalten 'lat', 'lng', 'city' und 'is_expensive'
# Konvertieren Sie den DataFrame in ein GeoDataFrame
# Erstellen Sie zunächst eine 'geometry'-Spalte mit Punkt-Daten
daten['geometry'] = daten.apply(lambda row: Point(row['lng'], row['lat']), axis=1)

# Konvertieren Sie nun den DataFrame in ein GeoDataFrame
# Setzen Sie das Koordinatenreferenzsystem (CRS) auf WGS84 (EPSG:4326)
gdf = gpd.GeoDataFrame(daten, geometry='geometry', crs='EPSG:4326')

# Reprojizieren Sie die Daten in Web Mercator (EPSG:3857) für die Verwendung mit contextily
gdf = gdf.to_crs(epsg=3857)

# Erstellen Sie eine Grafik für jede Stadt
for stadt in gdf['city'].unique():
    # Filtern Sie die Daten für die aktuelle Stadt
    stadt_daten = gdf[gdf['city'] == stadt]
    
    # Erstellen Sie eine Grafik
    fig, ax = plt.subplots(figsize=(10, 10))
    # Zeichnen Sie nicht teure Punkte
    nicht_teuer = stadt_daten[stadt_daten['is_expensive'] == False]
    ax.scatter(nicht_teuer.geometry.x, nicht_teuer.geometry.y, color='green', label='Nicht Teuer')
    teuer = stadt_daten[stadt_daten['is_expensive'] == True]
    ax.scatter(teuer.geometry.x, teuer.geometry.y, color='red', label='Teuer')
    # Zeichnen Sie nicht teure Punkte

    # Fügen Sie die Grundkarte hinzu
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    # Entfernen Sie die Achsen
    ax.set_axis_off()
    # Fügen Sie die Legende hinzu
    ax.legend()
    # Speichern Sie die Grafik in einer Datei
    
    #ax.set_title(f'Klassifikation von {stadt}', fontsize=15)
    plt.tight_layout()
    plt.savefig(f"Figures/Klassifikation/{stadt}.svg", format='svg')
    plt.show
