# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 11:42:13 2023

@author: Dennis
"""

import pandas as pd
# Annahme: Ihre Daten sind in einer CSV-Datei namens "data.csv"
path = "C:/Users/Dennis/Documents/Aemf1.csv"

data = pd.read_csv(path, sep=';', decimal=',')
filtered_data = data[data["Day"] != "Weekend"]


# Annahme: Sie möchten die gefilterten Daten in einer neuen Datei namens "filtered_data.csv" speichern
filtered_data.to_csv("C:/Users/Dennis/Documents/filtered_data_OnlyWeekday.csv",sep=';', decimal=',', index=False)

