# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:23:32 2024

@author: Dennis
"""

import pandas as pd
# Gaussian Mixture Models (GMM) Analysis
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np


df = pd.read_csv("/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv", sep=';', decimal='.', engine='python')



# Selecting numerical features for GMM
numerical_features = df.select_dtypes(include=[np.number])

# Standardizing the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)

# Defining the GMM
# We will start with a GMM with 5 components
# The number of components can be adjusted based on model performance and business understanding
gmm = GaussianMixture(n_components=5, random_state=42)

# Fitting the GMM
clusters = gmm.fit_predict(scaled_features)

# Adding the cluster labels to the original dataframe
df['cluster'] = clusters

# Displaying the first few rows with the cluster labels
print(df.head())


import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', data=df)
plt.title('Verteilung der Cluster')
plt.xlabel('Cluster')
plt.ylabel('Anzahl der Einträge')
plt.show()



from sklearn.decomposition import PCA

# Applying PCA for dimensionality reduction for visualization
pca = PCA(n_components=2)
scaled_features_pca = pca.fit_transform(scaled_features)

# Creating a new DataFrame for the PCA results
pca_df = pd.DataFrame(scaled_features_pca, columns=['PCA1', 'PCA2'])
pca_df['cluster'] = clusters

# Plotting the PCA results
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=pca_df, palette='Set1', alpha=0.8)
plt.title('PCA von GMM-Clustern')
plt.xlabel('Erste Hauptkomponente (PCA1)')
plt.ylabel('Zweite Hauptkomponente (PCA2)')
plt.legend(title='Cluster')
plt.show()

# Extracting the centroids of the clusters
# The centroids are the means of the Gaussians in the mixture
# Each centroid represents the "average" of the cluster
centroids = gmm.means_

# Inverse transform the centroids to get them back into the original feature space
original_space_centroids = scaler.inverse_transform(centroids)

# Creating a DataFrame for the centroids for easier interpretation
centroids_df = pd.DataFrame(original_space_centroids, columns=numerical_features.columns)

# Displaying the centroids
print(centroids_df)