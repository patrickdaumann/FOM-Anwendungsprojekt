#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 15:08:34 2023

@author: patrick


Ziele:
    - Aufteilen in Training und Testset

Versionsog:
    - V1.0 - 05.01.2023 - Ursprüngliche Version der PP Pipeline
"""

import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from scipy.stats import kstest

############################## Configuration
pipelineVersionNumber = 1.0

basePath = '/Users/patrick/GitHub/FOM-Anwendungsprojekt/' #trailing Slash!
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

# Index Reset -> Index soll kontinuierlich und lückenlos sein
df.reset_index(drop=True, inplace=True)


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


############################## Aufteilen ist Train, Test und Validation Sets

#Index des Dataframes zurücksetzen
df.reset_index(drop=True, inplace=True)

# Zufällige Aufteilung des DataFrames
frac_train = 0.7  # Prozentsatz für den Trainingsdatensatz
frac_test = 1 - frac_train  # Prozentsatz für den Testdatensatz (restliche Daten)

# DataFrame in Trainings- und Testdaten aufteilen
train_df = df.sample(frac=frac_train, random_state=42)  # Trainingsdaten
test_df = df.drop(train_df.index).reset_index(drop=True)  # Testdaten

test_df.reset_index(drop=True, inplace=True)

frac_test = 0.5
frac_val = 1 - frac_test

val_df = test_df.sample(frac=frac_val, random_state=42)
test_df = test_df.drop(val_df.index).reset_index(drop=True)  # Validationsdaten



############################## Übersicht der Daten
# Ausgabe von Informationen über den Dataframe in der Konsole
print(df.describe())
print(df.info())
print(df.head())

# Boxplot der Werte
df.boxplot(column='realSum', showfliers=False)

# Titel und Achsenbeschriftungen hinzufügen
plt.title('Boxplot für realSum')

# Diagramm anzeigen
plt.show()


############################## Export der Daten

# Export ja nein ?
if True:
    
    exportFilenamePrefix = 'Airbnb_Prices'
    exportFilenameSuffix = f"{pipelineVersionNumber}"
    exportFilenameExtension = '.csv'
    
    # Ganz
    exportFilenameFull = f"{exportFilenamePrefix}_V{exportFilenameSuffix}_Full{exportFilenameExtension}"

    
    # Aufgeteilte Sets für machinelles Lernen
    exportFilenameTrain = f"{exportFilenamePrefix}_V{exportFilenameSuffix}_Train{exportFilenameExtension}"
    exportFilenameTest = f"{exportFilenamePrefix}_V{exportFilenameSuffix}_Test{exportFilenameExtension}"
    exportFilenameVal = f"{exportFilenamePrefix}_V{exportFilenameSuffix}_Val{exportFilenameExtension}"

    
    #Kombinierten Dataframe exportieren
    exportFilePathFull = f"{exportBasePath}{exportFilenameFull}"
    print(f"Exportiere ganzen Datensatz nach: {exportFilePathFull}")
    df.to_csv(exportFilePathFull, index=False, sep=';')
    print("Fertig")
    
    #Trainingsdaten exportieren
    exportFilePathTrain = f"{exportBasePath}{exportFilenameTrain}"
    print(f"Exportiere Trainingsdatensatz nach: {exportFilePathTrain}")
    train_df.to_csv(exportFilePathTrain, index=False, sep=';')
    print("Fertig")
    
    #Testdaten exportieren
    exportFilePathTest = f"{exportBasePath}{exportFilenameTest}"
    print(f"Exportiere Testdatensatz nach: {exportFilePathTest}")
    test_df.to_csv(exportFilePathTest, index=False, sep=';')
    print("Fertig")
    
    #Validation Daten exportieren
    exportFilePathVal = f"{exportBasePath}{exportFilenameVal}"
    print(f"Exportiere Validationsdatensatz nach: {exportFilePathVal}")
    val_df.to_csv(exportFilePathVal, index=False, sep=';')
    print("Fertig")
    

############################ Verteilungsanalyse
# Ziel: P Value der Normalverteilung für alle metrischen Columns ermitteln, Histogramme abspeichern

# Ihre Daten auswählen
metric_data = df[['realSum', 'person_capacity', 'bedrooms', 'dist', 'metro_dist', 'attr_index', 'rest_index']]

for column in metric_data.columns:
    
    
    plt.figure(figsize=(6, 4))
    plt.hist(df[column], alpha=0.5, bins='auto')
    plt.xlabel('Werte')
    plt.ylabel('Häufigkeit')
    plt.title(f'Histogramm für {column}')
    plt.savefig(f"{exportBasePath}/Figures/{column}.pdf", format='pdf')
    plt.show()
    
    result = kstest(df[column], 'norm')

    # Ergebnisse in der Konsole ausgeben
    print(f'Kolmogorow-Smirnov-Test für {column}: Statistik={result.statistic:.4f}, p-Wert={result.pvalue:.4f}')
    

############################### Informationen über alle metrischen Spalten erfassen und als CSV ausgeben
stats_dict = {}
stats_list = []

for col in metric_data.columns:
    col_data = metric_data[col]
    stats = {
        'Spalte': col,
        'Median': col_data.median(),
        'Min': col_data.min(),
        'Max': col_data.max(),
        'Q1': np.percentile(col_data, 25),
        'Q3': np.percentile(col_data, 75),
        'IQR': np.percentile(col_data, 75) - np.percentile(col_data, 25),
        'Varianz': col_data.var(),
        'Standardabweichung': col_data.std(),
        'Spannweite': col_data.max() - col_data.min()
    }
    stats_list.append(stats)

# Liste in Dataframe wandeln
stats_df = pd.DataFrame(stats_list)

# Dateinamen und Pfade für Export
exportFilenameStats = f"{exportFilenamePrefix}_V{exportFilenameSuffix}_Stats{exportFilenameExtension}"
exportFilePathStats = f"{exportBasePath}{exportFilenameStats}"

# DataFrame in CSV exportieren
stats_df.to_csv(exportFilePathStats, index=False, sep=';')














