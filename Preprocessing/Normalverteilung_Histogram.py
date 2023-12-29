#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:18:50 2023

@author: patrick
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def makeHistogram(df, column: str, bins: int = 30, format='svg'):
    # Histogramm 
    plt.hist(x=df[column], bins=bins)  # Anzahl der Bins kann angepasst werden
    plt.xlabel(column)
    plt.ylabel('Häufigkeit')
    plt.savefig(f"/Users/patrick/GitHub/FOM_Anwendungsprojekt/projekt/figures/{column}_hist.{format}", format=format)
    plt.show()
    

def checkNorm(df, column: str):
    res = stats.cramervonmises(df['Price'], 'norm')
    print(f"Die Daten sind mit einem p von: {res.pvalue} Normalverteilt")
    
    

path = '/Users/patrick/GitHub/FOM_Anwendungsprojekt/projekt/Aemf1.csv'

df = pd.read_csv(path)

columnstocheck = [
    "Price",
    "Person Capacity",
    "Bedrooms",
    "City Center (km)",
    "Metro Distance (km)",
    "Attraction Index",
    "Normalised Attraction Index",
    "Restraunt Index",
    "Normalised Restraunt Index"
]

for column in columnstocheck:
    makeHistogram(df, column, 30, 'svg')
    checkNorm(df, column)
