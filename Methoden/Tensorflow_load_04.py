from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.utils import plot_model
import seaborn as sns


df = pd.read_csv('/mnt/c/Users/MK/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv', sep=';', decimal='.')
x = df[['room_type_encoded', 'rest_index_norm', 'metro_dist', 'dist', 'bedrooms', 'AttractionScore_Norm', 'city_encoded']].values
y = df[['realSum_Normalized']].values
maxrealSum = df[['realSum']].values.max()
#x = x.astype('float32')  # Konvertieren in float32

# Pfad zur H5-Datei
model_path = '/mnt/c/Users/MK/FOM-Anwendungsprojekt/Models/train-csv-r_t_e-r_i_n-m_d-d-b-AS_N-c_e-15kepochs_001.h5'

# Laden des Modells
model = keras.models.load_model(model_path)

# Modellübersicht anzeigen
model.summary()

# Vorhersagen treffen
predictions = model.predict(x)

# Listen für Ergebnisse
predictions_list = []
actual_list = []
delta_list = []

# Vorhersagen und Deltas sammeln
for i in range(len(y)):
    delta = abs(predictions[i] - y[i]) * maxrealSum
    predictions_list.append(predictions[i][0])
    actual_list.append(y[i][0])
    delta_list.append(delta[0])

# Erstellen eines DataFrames aus den Listen
results_df = pd.DataFrame({'Prediction': predictions_list, 'Actual': actual_list, 'Delta': delta_list})

plt.figure(figsize=(10,6))
sns.scatterplot(x=results_df['Actual'], y=results_df['Prediction'])
plt.xlabel('Tatsächliche Preise')
plt.ylabel('Vorhergesagte Preise (Tensorflow DL)')
plt.title('Tatsächliche vs. Vorhergesagte Preise (Tensorflow DL)')
plt.savefig(f"Figures/RandomForest/Preisvorhersage.svg", format='svg')
plt.show()

# Ergebnisse in CSV speichern
results_df.to_csv('/mnt/c/Users/MK/FOM-Anwendungsprojekt/Results/predictions_02.csv', index=False)

# Durchschnittliches Delta berechnen
average_delta = results_df['Delta'].mean()
print("Durchschnittliches Delta:", average_delta)