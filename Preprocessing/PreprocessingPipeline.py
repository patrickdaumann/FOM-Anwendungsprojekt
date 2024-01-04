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

import os, re
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder

############################## Configuration
basePath = '/Users/patrick/GitHub/FOM-Anwendungsprojekt/'#trailing Slash!
exportBasePath = f"{basePath}Data/Output/" #trailing Slash!

# Verzeichnis mit einzelnen CSV Dateien
sourcefilepath = f"{basePath}Data/Source/Airbnb_Prices_in_European_Cities"

#Erzeugen eines leeren Dataframes
df = pd.DataFrame()

# Alle einzelnen CSV Dateien zu einem Dataframe zusammenführen 
# dabei die Information Weekday und City mit einbeziehen.
if os.path.isdir(sourcefilepath):
    
    # Iterieren durch die Dateien im Quellverzeichnis
    for filename in os.listdir(sourcefilepath):
        
        #Zusammensetzen des vollen Dateipfades aus Quellverzeichnis und Dateiname
        fullfilepath = f"{sourcefilepath}/{filename}"
        
        #Einlesen des Temp Dataframe aus neuer Datei
        dftemp = pd.read_csv(fullfilepath)
        
        #Auslesen von daytype und city aus Dateinamen mit regex
        muster = r"([a-zA-Z]+)_([a-zA-Z]+).csv"
        treffer = re.match(muster, filename)
        if treffer:
            dftemp['city'] = treffer.group(1)
            dftemp['daytype'] = treffer.group(2)
        
        #Anfügen des temp Dataframe
        df = pd.concat([df, dftemp])
        

# Entfernen des Index aus den einzelnen csv Dateien    
df = df.drop(columns=['Unnamed: 0'])


############################ Auf fehlende Werte prüfen
missing = df.isna().sum()
if missing.any():
    print(f"fehlende Datensätze: {missing}")

else:
    print("keine fehlenden Datensätze (NaN oder NULL)")

############################ Auf Duplikate prüfen
dups = df.duplicated().sum()
if dups.any():
    print("Es wurden duplikate gefunden!")
    df.drop_duplicates()
else:
    print("Es wurden keine duplikate gefunden!")

############################ Kategorische Informationen Kodieren

categorical_columns = df.select_dtypes(include=['object']).columns
#print("Categorical columns:", categorical_columns)

le = LabelEncoder()

for categorical_column in categorical_columns:
    df[f"{categorical_column}_encoded"] = le.fit_transform(df[categorical_column])


#print(df['room_type_encoded'])

############################ Boolische Werte in 0 & 1 konvertieren

boolean_columns = ['room_shared','room_private','host_is_superhost']

for column in boolean_columns:
    df[column] = df[column].astype(int)

############################ Bereinigung der Werte

column = 'realSum'
bins = 100

Bereinigungsmethode = 'Z-SCORE' #'IQR', 'Z-SCORE'

if Bereinigungsmethode == 'IQR':
    # Bereinigen der Spalte realSum mit der IQR-Methode:
    Q1 = df['realSum'].quantile(0.25)
    Q3 = df['realSum'].quantile(0.75)
    IQR = Q3 - Q1
    mad = stats.median_absolute_deviation(df['realSum'])
    
    # Schwellenwert für Ausreißer (zum Beispiel, 1.5*IQR + 3*MAD)
    threshold = 1.5 * IQR + 3 * mad
    
    # Filtern der Ausreißer
    df = df[(df['realSum'] < Q3 + threshold) & (df['realSum'] > Q1 - threshold)]
    
    
    # Histogramm der bereinigten Daten ausgeben
    plt.hist(x=df[column], bins='auto')
    plt.xlabel(column)
    plt.ylabel('Häufigkeit')
    plt.show()

elif True:
    z_scores = stats.zscore(df['realSum'])
    threshold = 3
    df = df[(z_scores < threshold)]


# Boxplot der bereinigten Werte
df.boxplot(column='realSum', showfliers=False)

# Titel und Achsenbeschriftungen hinzufügen
plt.title('Boxplot für realSum')

# Diagramm anzeigen
plt.show()


####################### Normalisierung der Daten

min_value = df['realSum'].min()
max_value = df['realSum'].max()

df['realSum_Normalized'] = (df['realSum'] - min_value) / (max_value - min_value)


# Histogramm der Normalisierten Daten ausgeben
plt.hist(x=df['realSum_Normalized'], bins='auto')
plt.xlabel(column)
plt.ylabel('Häufigkeit')
plt.show()

#BoxPlot
df.boxplot(column='realSum_Normalized', showfliers=True)
plt.title('Boxplot für realSum_Normalized')
plt.show()


############################## Übersicht der Daten
# Ausgabe von Informationen über den Dataframe in der Konsole
print(df.describe())
print(df.info())
print(df.head())


############################## Export der Daten

# Ganzer Datensatz

if True:
    # Ganz
    exportFilenameFull = 'Airbnb_Prices_Full.csv'

    
    # Aufgeteilt für ML
    exportFilenameTrain = 'Airbnb_Prices_Train.csv'
    exportFilenameTest = 'Airbnb_Prices_Test.csv'

    
    #Kombinierten Dataframe exportieren
    exportFilePathFull = f"{exportBasePath}{exportFilenameFull}"
    print(f"Exportiere ganzen Datensatz nach: {exportFilePathFull}")
    df.to_csv(exportFilePathFull, index=False, sep=';')
    print("Fertig")
    



    



















