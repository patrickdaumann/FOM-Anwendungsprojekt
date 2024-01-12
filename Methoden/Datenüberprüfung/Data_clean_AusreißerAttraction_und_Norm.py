import pandas as pd
from scipy import stats
import numpy as np
# Load the data
file_path = '/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/AttractionScores_cleaned.csv'
data = pd.read_csv(file_path, delimiter=';', decimal='.')

# Display the head of the dataframe
print(data.head())



# Identify and remove outliers in 'AttractionScore' using z-score
z_scores = stats.zscore(data['realSum'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3) # typically, a z-score of 3 is considered to be an outlier

# Filter the data
data_clean = data[filtered_entries]

# Normalize the 'AttractionScore' column
max_score = data_clean['realSum'].max()
min_score = data_clean['realSum'].min()
data_clean['realSum_Normalized'] = (data_clean['realSum'] - min_score) / (max_score - min_score)

# Display the head of the cleaned dataframe
print(data_clean.head())


data_clean.to_csv('C:/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/AttractionScoresRealSum_cleaned.csv', sep=';', decimal='.', index=False)