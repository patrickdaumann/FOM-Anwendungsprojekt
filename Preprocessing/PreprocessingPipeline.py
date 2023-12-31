#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 15:08:34 2023

@author: patrick


Ziele:
    - Zusammenfügen der CSV Dateien aus dem Ursprünglichen Dataset
    - Duplikate Testen
    - Fehlende Werte testen -> Zählen und Visuell aufbereiten
    - Normalisieren der Werte wenn Sinnvoll
    - Ausreißerbehandlung
    	- Rausfiltern? 
    - Aufteilen in Training und Testset
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

exportCSV = False

# Verzeichnis mit einzelnen CSV Dateien
sourcefilepath = '/Users/patrick/GitHub/FOM-Anwendungsprojekt/Data/Source/Airbnb_Prices_in_European_Cities'

# Verzeichnis für den Export des bearbeiteten Dataframes als CSV
exportfilepath = '/Users/patrick/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Combined.csv'

#Erzeugen eines leeren Dataframes
df = pd.DataFrame()

# Alle einzelnen CSV Dateien zu einem Dataframe zusammenführen
if os.path.isdir(sourcefilepath):
    
    # Iterieren durch die Dateien im Quellverzeichnis
    for filename in os.listdir(sourcefilepath):
        
        #Zusammensetzen des vollen Dateipfades aus Quellverzeichnis und Dateiname
        fullfilepath = f"{sourcefilepath}/{filename}"
        
        #Einlesen des Temp Dataframe aus neuer Datei
        dftemp = pd.read_csv(fullfilepath)
        
        #Anfügen des temp Dataframe
        df = pd.concat([df, dftemp])
        

# Entfernen des Index aus den einzelnen csv Dateien    
df = df.drop(columns=['Unnamed: 0'])

# Ausgaabe von Informationen über den Dataframe in der Konsole
#print(df.describe())
#print(df.info())
#print(df.head())

if exportCSV == True:
    #Kombinierten Dataframe exportieren
    df.to_csv(exportfilepath, index=False, sep=';')



############################ Bereinigung Tests
column = 'realSum'
bins = 100


# Bereinigen der Spalte realSum mit der IQR-Methode:
Q1 = df['realSum'].quantile(0.25)
Q3 = df['realSum'].quantile(0.75)
IQR = Q3 - Q1
mad = stats.median_absolute_deviation(df['realSum'])

# Schwellenwert für Ausreißer (zum Beispiel, 1.5*IQR + 3*MAD)
threshold = 1.5 * IQR + 3 * mad

# Filtern Sie Ausreißer und erstellen Sie einen bereinigten DataFrame
cleaned_df = df[(df['realSum'] < Q3 + threshold) & (df['realSum'] > Q1 - threshold)]


plt.hist(x=cleaned_df[column], bins='auto')
plt.xlabel(column)
plt.ylabel('Häufigkeit')
plt.show()

print(cleaned_df.describe())
print(cleaned_df.info())


# Boxplot der bereinigten Werte
cleaned_df.boxplot(column='realSum', showfliers=False)

# Titel und Achsenbeschriftungen hinzufügen
plt.title('Boxplot für realSum')

# Diagramm anzeigen
plt.show()



















