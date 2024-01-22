# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:22:36 2024

@author: Dennis
"""

import pandas as pd
import numpy as np
from IPython.display import display


import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

# Laden des Datensatzes
pfad = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_full.csv"
df = pd.read_csv(pfad, sep=';', decimal='.' , low_memory=False)

ausgeschlossene_spalten = ['attr_index', 'attr_index_norm', 'lat', 'lng', 'realSum',
                    'room_type', 'room_shared', 'room_private', 'daytype',
                    'city_encoded', 'daytype_encoded', 'AttractionScore', 'city']
df = df.drop(columns=ausgeschlossene_spalten)

print('First 5 rows:')
display(df.head())

corr_matrix = df.corr()
print('Correlation matrix:')
display(corr_matrix)

print('Done')

import seaborn as sns
import matplotlib.pyplot as plt

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.tight_layout()
plt.savefig(f"Figures/Korrelationsmatrix.svg", format='svg')
plt.show()