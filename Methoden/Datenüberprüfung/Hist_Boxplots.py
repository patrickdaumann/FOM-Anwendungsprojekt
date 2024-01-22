# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 09:57:32 2024

@author: Dennis
"""

import pandas as pd

# Aufteilen der Daten in separate Spalten
airbnb_data = pd.read_csv("/mnt/c/Users/MK/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv", sep=';',decimal='.', engine='python')

# Entfernen von führenden und nachfolgenden Leerzeichen in Spaltennamen
airbnb_data.columns = [c.strip() for c in airbnb_data.columns]

# Anzeigen der ersten Zeilen der Tabelle nach der Aufteilung
print(airbnb_data.head())
# Beschreibung der Daten
print(airbnb_data.describe(include='all'))



# Überprüfung auf fehlende Werte
missing_values = airbnb_data.isnull().sum()
print('Fehlende Werte pro Spalte:\n', missing_values)

# Überprüfung auf Ausreißer in den numerischen Spalten
numerical_columns = airbnb_data.select_dtypes(include=['float64', 'int64']).columns
print('Statistische Zusammenfassung der numerischen Spalten:')
for column in numerical_columns:
    print(f'\nStatistik f\u00fcr {column}:')
    print(airbnb_data[column].describe())
    
    
    
import matplotlib.pyplot as plt
import seaborn as sns

# Einstellen des Stils für die Plots
sns.set(style='whitegrid')

# Erstellen von Histogrammen für numerische Spalten
for column in numerical_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(airbnb_data[column], kde=True)
    plt.title('Histogramm von ' + column)
    plt.xlabel(column)
    plt.ylabel('Häufigkeit')
    plt.show()

# Erstellen von Boxplots für numerische Spalten
for column in numerical_columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=airbnb_data[column])
    plt.title('Boxplot von ' + column)
    plt.xlabel(column)
    plt.show()