# -*- coding: utf-8 -*-
"""
Erstellt am Dienstag, 16. Januar 2024

Autor: Dennis
"""
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Laden des Datensatzes
pfad = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_full.csv"
daten = pd.read_csv(pfad, sep=';', decimal='.' , low_memory=False)

# Löschen der angegebenen Spalten
spalten_zum_löschen = ['attr_index', 'attr_index_norm', 'lat', 'lng', 'realSum',
                   'room_type', 'room_shared', 'room_private', 'daytype',
                   'city_encoded', 'daytype_encoded', 'AttractionScore']
daten_reduziert = daten.drop(columns=spalten_zum_löschen)

# Auswahl nur numerischer Daten
numerische_daten = daten_reduziert.select_dtypes(include=['float64', 'int64'])

# Standardisieren der Daten
scaler = StandardScaler()
numerische_daten_skaliert = scaler.fit_transform(numerische_daten)

# Anwenden von t-SNE
tsne = TSNE(n_components=2, random_state=42)
numerische_daten_tsne = tsne.fit_transform(numerische_daten_skaliert)

# Konvertieren der t-SNE-Ausgabe in ein DataFrame
tsne_df = pd.DataFrame(numerische_daten_tsne, columns=['TSNE1', 'TSNE2'])

# Anzeigen des Kopfs des t-SNE-Dataframes
print(tsne_df.head())

tsne_df.to_csv('Figures/MM_Kmeans/tsne_df.csv', sep=';', decimal='.', index=False)


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer

from sklearn.metrics import silhouette_score

# Wir werden Dictionaries erstellen, um die optimale Anzahl der Cluster für jede Stadt zu speichern
optimal_clusters_elbow = {}
optimal_clusters_silhouette = {}

# Gruppieren der t-SNE-Daten nach Stadt
tsne_df['city'] = daten['city']
stadtgruppen = tsne_df.groupby('city')

# Bestimmung der optimalen Anzahl der Cluster für jede Stadt mithilfe der Ellbogenmethode
for stadt, gruppe in stadtgruppen:
    # Entfernen der Stadtspalte für das Clustering
    gruppendaten = gruppe.drop(columns=['city'])
    
    # Initialisieren des KMeans-Modells für die Ellbogenmethode
    modell_ellbogen = KMeans()
    visualizer_ellbogen = KElbowVisualizer(modell_ellbogen, k=(2,10), timings=False,)
    visualizer_ellbogen.fit(gruppendaten)
    optimal_clusters_elbow[stadt] = visualizer_ellbogen.elbow_value_
    visualizer_ellbogen.ax.set_title(f"Ellbogenmethode für optimale Cluster in {stadt}")
    visualizer_ellbogen.show(outpath=f"Figures/MM_Kmeans/elbow_{stadt}.svg")
    visualizer_ellbogen.show()

for stadt, gruppe in stadtgruppen:
    # Entfernen der Stadtspalte für das Clustering
    gruppendaten = gruppe.drop(columns=['city'])
    
    best_score = -1
    best_k = 2

    # Testen verschiedener Werte für k
    for k in range(2, 11):
        modell = KMeans(n_clusters=k, random_state=42)
        labels = modell.fit_predict(gruppendaten)
        score = silhouette_score(gruppendaten, labels)

        if score > best_score:
            best_score = score
            best_k = k

    optimal_clusters_silhouette[stadt] = best_k

    # Optional: Plot der Silhouettenwerte für verschiedene k-Werte
    plt.figure()
    plt.plot(range(2, 11), [silhouette_score(gruppendaten, KMeans(n_clusters=k, random_state=42).fit_predict(gruppendaten)) for k in range(2, 11)])
    plt.title(f"Silhouettenwerte für verschiedene k-Werte - {stadt}")
    plt.xlabel("Anzahl der Cluster (k)")
    plt.ylabel("Silhouettenwert")
    plt.savefig(f"Figures/MM_Kmeans/silhouette_{stadt}.svg", format='svg')
    plt.show()

print('Optimale Anzahl der Cluster für jede Stadt mithilfe der Ellbogenmethode:')
print(optimal_clusters_elbow)

print('Optimale Anzahl der Cluster für jede Stadt mithilfe des Silhouettenwerts:')
print(optimal_clusters_silhouette)

# Erstellen eines Dictionaries, um die Clusterzentren für jede Stadt zu speichern
clusterzentren = {}

# Durchführung des K-Means-Clustering und Plotten für jede Stadt
for stadt, cluster in optimal_clusters_elbow.items():
    # Auswahl der Daten für die aktuelle Stadt
    stadt_daten = tsne_df[tsne_df['city'] == stadt].drop(columns=['city'])
    
    # Durchführung des K-Means-Clustering
    kmeans = KMeans(n_clusters=cluster, random_state=42)
    stadt_daten['cluster'] = kmeans.fit_predict(stadt_daten)
    
    # Speichern der Clusterzentren
    clusterzentren[stadt] = kmeans.cluster_centers_
    
    # Plotten der Cluster
    plt.figure(figsize=(8, 6))
    plt.scatter(stadt_daten.iloc[:, 0], stadt_daten.iloc[:, 1], c=stadt_daten['cluster'], cmap='viridis', marker='o')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
    plt.title('K-Means-Clustering für ' + stadt)
    plt.xlabel('tSNE 1')
    plt.ylabel('tSNE 2')
    plt.savefig(f"Figures/MM_Kmeans/Centers_elbow_{stadt}.svg", format='svg')
    plt.show()

print('Clusterzentren für jede Stadt:')
print(clusterzentren)

# Konvertieren des Clusterzentren-Dictionaries in einzelne Dataframes und dann Zusammenführen
clusterzentren_dfs = []
for stadt, zentren in clusterzentren.items():
    # Konvertieren der Clusterzentren jeder Stadt in ein Dataframe
    stadt_df = pd.DataFrame(zentren, columns=[f'PC{i+1}' for i in range(zentren.shape[1])])
    stadt_df['city'] = stadt
    clusterzentren_dfs.append(stadt_df)

# Zusammenführen aller einzelnen Stadt-Dataframes
final_cluster_centers_df = pd.concat(clusterzentren_dfs, ignore_index=True)

# Exportieren des Dataframes in eine CSV-Datei
final_cluster_centers_df.to_csv('Figures/MM_Kmeans/final_cluster_centers_elbow_PCA.csv', index=False, sep=';', decimal='.')

print('Clusterzentren wurden in final_cluster_centers.csv exportiert.')


# Erstellen eines Dictionaries, um die Clusterzentren für jede Stadt zu speichern
clusterzentren = {}

# Durchführung des K-Means-Clustering und Plotten für jede Stadt
for stadt, cluster in optimal_clusters_silhouette.items():
    # Auswahl der Daten für die aktuelle Stadt
    stadt_daten = tsne_df[tsne_df['city'] == stadt].drop(columns=['city'])
    
    # Durchführung des K-Means-Clustering
    kmeans = KMeans(n_clusters=cluster, random_state=42)
    stadt_daten['cluster'] = kmeans.fit_predict(stadt_daten)
    
    # Speichern der Clusterzentren
    clusterzentren[stadt] = kmeans.cluster_centers_
    
    # Plotten der Cluster
    plt.figure(figsize=(8, 6))
    plt.scatter(stadt_daten.iloc[:, 0], stadt_daten.iloc[:, 1], c=stadt_daten['cluster'], cmap='viridis', marker='o')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
    plt.title('K-Means-Clustering für ' + stadt)
    plt.xlabel('tSNE 1')
    plt.ylabel('tSNE 2')
    plt.savefig(f"Figures/MM_Kmeans/Centers_silhouette_{stadt}.svg", format='svg')
    plt.show()

print('Clusterzentren für jede Stadt:')
print(clusterzentren)

# Konvertieren des Clusterzentren-Dictionaries in einzelne Dataframes und dann Zusammenführen
clusterzentren_dfs = []
for stadt, zentren in clusterzentren.items():
    # Konvertieren der Clusterzentren jeder Stadt in ein Dataframe
    stadt_df = pd.DataFrame(zentren, columns=[f'PC{i+1}' for i in range(zentren.shape[1])])
    stadt_df['city'] = stadt
    clusterzentren_dfs.append(stadt_df)

# Zusammenführen aller einzelnen Stadt-Dataframes
final_cluster_centers_df = pd.concat(clusterzentren_dfs, ignore_index=True)

# Exportieren des Dataframes in eine CSV-Datei
final_cluster_centers_df.to_csv('Figures/MM_Kmeans/final_cluster_centers_MM.csv', index=False, sep=';', decimal='.')

print('Clusterzentren wurden in final_cluster_centers_MM.csv exportiert.')
