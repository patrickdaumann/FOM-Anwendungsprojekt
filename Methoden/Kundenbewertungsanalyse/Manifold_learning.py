# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 13:58:13 2024

@author: Dennis
"""

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv", sep=';', decimal='.', engine='python')
# Since the categorical variables are already encoded, we can proceed with the analysis


# Auswahl der numerischen Spalten f�r das Manifold Learning
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Skalierung der numerischen Daten
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[numerical_columns])

# Anwendung von t-SNE
# Da t-SNE rechenintensiv sein kann, beschr�nken wir die Anzahl der Zeilen f�r dieses Beispiel
# und verwenden tqdm, um den Fortschritt anzuzeigen
# Entfernen von tqdm, da es zu einem Fehler f�hrte

# Anwendung von t-SNE auf die ersten 1000 skalierten Datenpunkte
tsne = TSNE(n_components=2, verbose=1, random_state=42)
tsne_results = tsne.fit_transform(scaled_features[:1000])

# Visualisierung der Ergebnisse
df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
plt.figure(figsize=(16,10))
plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'])
plt.title('t-SNE Ergebnisse')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.show()


from sklearn.cluster import KMeans

# Anwendung des K-Means-Clustering-Algorithmus
# Die Anzahl der Cluster wird zun�chst willk�rlich auf 5 gesetzt
kmeans = KMeans(n_clusters=5, random_state=42)
df_tsne['cluster'] = kmeans.fit_predict(tsne_results)

# Visualisierung der Cluster
plt.figure(figsize=(16,10))
plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'], c=df_tsne['cluster'], cmap='viridis')
plt.title('t-SNE Ergebnisse mit K-Means-Clustering')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.colorbar()
plt.show()


# Profiling der Cluster
cluster_profiles = df[numerical_columns].groupby(df_tsne['cluster']).mean()

# Anzeige der durchschnittlichen Werte der numerischen Merkmale f�r jedes Cluster
print(cluster_profiles)



# Pr�fung, ob die 'cluster' Spalte in 'df' existiert und korrekt benannt ist
print(df.columns)

# Falls 'df_tsne' die 'cluster' Spalte enth�lt, f�gen wir diese zu 'df' hinzu
if 'cluster' in df_tsne.columns:
    df['cluster'] = df_tsne['cluster']
    print('Cluster column added to df.')
else:
    print('Cluster column not found in df_tsne.')
    
    
    # Analyse der Preisverteilung innerhalb der Cluster
price_distribution_per_cluster = df[['realSum', 'cluster']].groupby('cluster').describe()

# Anzeige der Preisverteilung
print(price_distribution_per_cluster)



# Analyse der Kundenzufriedenheit innerhalb der Cluster
satisfaction_per_cluster = df[['guest_satisfaction_overall', 'cluster']].groupby('cluster').describe()

# Visualisierung der Kundenzufriedenheit pro Cluster
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(x='cluster', y='guest_satisfaction_overall', data=df)
plt.title('Customer Satisfaction Scores per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Guest Satisfaction Overall')
plt.show()



# Correlation analysis between price and satisfaction within clusters
import numpy as np

# Calculate the correlation matrix
correlation_matrix = df[['realSum', 'guest_satisfaction_overall', 'cluster']].groupby('cluster').corr().reset_index()

# Filter out the correlation between the same variable
# We are interested in the correlation between 'realSum' and 'guest_satisfaction_overall'
correlation_matrix = correlation_matrix[correlation_matrix['level_1'] == 'realSum'][['cluster', 'guest_satisfaction_overall']]

# Rename columns for clarity
correlation_matrix.columns = ['Cluster', 'Price-Satisfaction Correlation']

# Display the correlation matrix
print(correlation_matrix)



# Further analysis: Exploring other variables that might influence customer satisfaction
# We will look at the relationship between room type, host status, and customer satisfaction.

# Calculate the average satisfaction score for each room type and host status
room_type_satisfaction = df.groupby('room_type')['guest_satisfaction_overall'].mean().reset_index()
host_status_satisfaction = df.groupby('host_is_superhost')['guest_satisfaction_overall'].mean().reset_index()


# Visualize the relationship
plt.figure(figsize=(14, 7))

# Room type satisfaction
plt.subplot(1, 2, 1)
sns.barplot(x='room_type', y='guest_satisfaction_overall', data=room_type_satisfaction)
plt.title('Average Customer Satisfaction by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Average Satisfaction Score')

# Host status satisfaction
plt.subplot(1, 2, 2)
sns.barplot(x='host_is_superhost', y='guest_satisfaction_overall', data=host_status_satisfaction)
plt.title('Average Customer Satisfaction by Host Status')
plt.xlabel('Is Superhost')
plt.ylabel('Average Satisfaction Score')



plt.tight_layout()
plt.show()