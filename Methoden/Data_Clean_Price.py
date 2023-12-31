# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 09:57:23 2023

@author: Dennis
"""

import pandas as pd

# Pfad zur CSV-Datei
path = 'C:/Users/Dennis/Documents/filtered_data_All.csv'

# Lese die CSV-Datei in ein pandas DataFrame ein
data = pd.read_csv(path, sep=';', decimal=',')

# Entferne Zeilen, in denen die 'price'-Spalte über 5000 liegt
data = data[data['Price'] <= 1000]

# Speichere das bereinigte DataFrame wieder in eine CSV-Datei
data.to_csv('C:/Users/Dennis/Documents/Aemf1_cleaned_1000.csv', sep=';', decimal=',', index=False)