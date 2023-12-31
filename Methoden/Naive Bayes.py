import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns




# Annahme: Ihre Daten sind in einer Datei mit dem Namen "data.csv"
path = "C:/Users/Dennis/Documents/Aemf1_cleaned_5000.csv"

data = pd.read_csv(path, sep=';', decimal=',')

data["Original_Guest Satisfaction"] = data["Guest Satisfaction"]

# Definieren Sie die Zielklasse: Hier teilen wir die "Price"-Spalte in teuer (1) und nicht teuer (0)
bestimmter_Schwellenwert =95  # Ihr Schwellenwert hier
data['Guest Satisfaction'] = data['Guest Satisfaction'].apply(lambda x: 1 if x >= bestimmter_Schwellenwert else 0)

# Verwenden Sie One-Hot-Encoding für alle relevanten kategorischen Merkmale
data_encoded = pd.get_dummies(data, columns=["Room Type", "City", "Day"])

# Trennen Sie die Daten in unabhängige Variablen (X) und die Zielvariable (y)
X = data_encoded.drop("Guest Satisfaction", axis=1)
y = data_encoded["Guest Satisfaction"]

# Aufteilen der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisieren des Naive-Bayes-Klassifikators (Gaussian Naive Bayes)
nb_classifier = GaussianNB()

# Anpassen des Klassifikators an die Trainingsdaten
nb_classifier.fit(X_train, y_train)

all_predictions = nb_classifier.predict(X)

# Vorhersagen auf den Testdaten
y_pred = nb_classifier.predict(X_test)

# Berechnen der Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualisierung der Confusion Matrix als Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Vorhergesagte Werte")
plt.ylabel("Tatsächliche Werte")
plt.title("Confusion Matrix")
plt.show()

# Berechnung und Anzeige von Genauigkeit, Präzision, Recall und F1-Score
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Genauigkeit (Accuracy): {accuracy}")
print("\nKlassifikationsbericht:")
print(class_report)




# Erstellen Sie eine neue Spalte "Teuer" oder "Nicht Teuer" basierend auf den Vorhersagen
data["Teuer"] = ["Teuer" if prediction == 1 else "Nicht Teuer" for prediction in all_predictions]

# Speichern Sie die Datenframe mit den Vorhersagen in eine CSV-Datei
data.to_csv("C:/Users/Dennis/Documents/Predicted_Apartments.csv", index=False, sep=';', decimal=',')