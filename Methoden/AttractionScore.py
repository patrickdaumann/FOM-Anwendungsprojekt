#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:01:39 2024

@author: patrick

Berechnung der Entfernung zwischen dem Apartment und 
"""

from geopy.distance import geodesic
import pandas as pd

# import Dataframe
fullfilepath = "/Users/patrick/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Full.csv"

df = pd.read_csv(fullfilepath, sep=';')

#Import Attractions.csv
attracttionsCSVPath = '/Users/patrick/GitHub/FOM-Anwendungsprojekt/Data/Source/Attractions/Attractions.csv'

attractions_df = pd.read_csv(attracttionsCSVPath, sep=';')

#Staedtenamen in Lowercase wandeln
attractions_df['City'] = attractions_df['City'].str.lower()

previous_city = None

for index, row in df.iterrows():    
    
    #SubDF mit der passenden Stadt erzeugen
    city = row['city']
    
    if previous_city != city:
        subdf = attractions_df.query('City == @city')
    
    #Geoloc Daten des Listings in Variable Speichern
    listingGeoLoc = (row['lat'], row['lng'])
    
    #Platzhalter Variablen    
    summe = 0
    maxrating = 0
    
    #Leeres Array für Tupel erstellen
    distances = []
    
    #Iterieren des subdf und berechnungen durchführen (Iteration durch alle Attractions für die passende Stadt)
    for indexy, rowy in subdf.iterrows():
        
        #Distanz berechnen
        dist = round(geodesic(listingGeoLoc, (rowy['lat'], rowy['lng'])).kilometers, 2)
        ratings = rowy['ratings']
        attractionname = rowy['Attraction']
        
        # Distanz und Ratinganzahl in Array abspeichern
        distances.append((dist, ratings, attractionname))
        
    #maxrating erfassen
    for dist, ratings, attractionname in distances:
        if maxrating < ratings:
            maxrating = ratings
    
    
    #Formel für Score: (Bewertung/maxBewertung) * (100/Distanz zur Attraktion) und das summiert für jede Attraktion
    AttractionScore = 0
    for dist, ratings, attractionname in distances:
        AttractionScore += (ratings/maxrating) * (100/dist)
    
    #Gerundeten Wert in Konsole ausgeben
    print(round(AttractionScore, 2))
    
    #Wert in den DF am passenden Index einfuegen
    df.at[index, 'AttractionScore'] = AttractionScore
    
    previous_city = row['city']
    
    
# Werte Normalisieren
AttracttionScoreMin = df['AttractionScore'].min()
AttracttionScoreMax = df['AttractionScore'].max()

df['AttractionScore_Norm'] = 100 * (df['AttractionScore'] - AttracttionScoreMin) / (AttracttionScoreMax - AttracttionScoreMin)


# DF Exportieren
exportpath = '/Users/patrick/GitHub/FOM-Anwendungsprojekt/Data/Output/AttractionScores.csv'
df.to_csv(exportpath, index=False, sep=';')































