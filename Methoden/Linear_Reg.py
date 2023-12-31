import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Annahme: Ihre Daten sind in einer Datei mit dem Namen "data.csv"
path = "C:/Users/Dennis/Documents/Aemf1_cleaned_5000.csv"

data = pd.read_csv(path, sep=';', decimal=',')

# Entfernen Sie die "City"-Spalte aus dem DataFrame "data"
data = data.drop("City", axis=1)
data = data.drop("Day", axis=1)
data = data.drop("Attraction Index", axis=1)
data = data.drop("Restraunt Index", axis=1)

# Konvertieren Sie kategorische Spalten in numerische mit One-Hot-Encoding
data_encoded = pd.get_dummies(data, columns=["Room Type"])

# Definieren Sie erneut abhängige und unabhängige Variablen
X = data_encoded.drop("Guest Satisfaction", axis=1)
y = data_encoded["Guest Satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Erstellen Sie Streudiagramme für jede unabhängige Variable gegen "Price"
for column in X.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(X[column], y, alpha=0.5)
    plt.xlabel(column)
    plt.ylabel("Guest Satisfaction")
    plt.title(f"Guest Satisfaction vs. {column}")
    plt.grid(True)
    plt.show()

# Erstellen Sie eine Korrelationsmatrix
correlation_matrix = data_encoded.corr()

plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
plt.title("Korrelationsmatrix")
plt.show()
