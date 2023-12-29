#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:24:47 2023

@author: patrick
"""

import pandas as pd
from scipy import stats
import numpy as np

def describeMetricColumn(df, column: str):
    """
    Python Funktion zur Beschreibung von Metrischen Daten

    Die Funktion erhält einen Pandas Dataframe und einen Spaltennamen und erzeugt ein Dictionary mit einer statistischen Beschreibung der Werte    

    :param df: Der Dataframe, der die zu Beschreibenden Werte enthält
    :type df: pandas.core.frame.DataFrame
    :param column: Der Spaltenname der zu beschreibenden Spalte 
    :type column: str
    :return: Dictionary Opbject mit der Beschreibung.
    :rtype: dict
    """
    
    describe_Dict = {}
    describe_Dict['Name'] = column
    describe_Dict['Median'] = np.median(df[column])
    describe_Dict['q75'] = np.percentile(df[column], 75)
    describe_Dict['q25'] = np.percentile(df[column], 25)
    describe_Dict['Mittelwert'] = np.mean(df[column])
    describe_Dict['Varianz'] = np.var(df[column])
    describe_Dict['Standardabweichung'] = np.std(df[column])
    describe_Dict['Min'] = np.min(df[column])
    describe_Dict['Max'] = np.max(df[column])
    describe_Dict['Standardabweichung'] = np.max(df[column]) 
    describe_Dict['Modus'] = stats.mode(df[column]).mode[0]
    
    
    return describe_Dict

path = '/Users/patrick/GitHub/FOM_Anwendungsprojekt/projekt/Aemf1.csv'

df = pd.read_csv(path)

x = describeMetricColumn(df, 'Price')

print(x)

    