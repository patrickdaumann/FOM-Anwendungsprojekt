# Einteilung der Zufriedenheitsbewertungen in Kategorien
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv("/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Full.csv", sep=';',decimal='.', engine='python')

# Wir definieren einen Schwellenwert für hohe Zufriedenheit
threshold = data['guest_satisfaction_overall'].mean()
data['satisfaction_category'] = (data['guest_satisfaction_overall'] >= threshold).astype(int)

# Auswahl einiger Merkmale für die Klassifikation
features = ['room_type', 'room_shared', 'room_private', 'person_capacity', 'host_is_superhost', 'multi', 'biz', 'cleanliness_rating', 'bedrooms', 'dist', 'metro_dist', 'attr_index', 'attr_index_norm', 'rest_index', 'rest_index_norm', 'lng', 'lat']

# Konvertierung von kategorischen Variablen in numerische Werte
for feature in features:
    if data[feature].dtype == 'object':
        data[feature] = data[feature].astype('category').cat.codes

# Aufteilung der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(data[features], data['satisfaction_category'], test_size=0.2, random_state=42)

# Anwendung des Naive Bayes-Klassifikators
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Vorhersagen auf dem Testset
y_pred = nb_classifier.predict(X_test)

# Bewertung der Leistung des Modells
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print('Accuracy of the Naive Bayes classifier:', accuracy)
print('Classification report:\n', report)


# Visualisierung der Ergebnisse der Naive Bayes Klassifikation
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Berechnung der Konfusionsmatrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualisierung der Konfusionsmatrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title('Confusion Matrix for Naive Bayes Classifier')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()