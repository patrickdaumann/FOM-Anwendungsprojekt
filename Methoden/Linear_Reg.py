import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Annahme: Ihre Daten sind in einer Datei mit dem Namen "data.csv"
path = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Train.csv"

path2 = "/Users/Dennis/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_Full_V1.0_Test.csv"

data = pd.read_csv(path, sep=';', decimal='.')
data_test = pd.read_csv(path2, sep=';', decimal='.')


# Konvertieren Sie kategorische Spalten in numerische mit One-Hot-Encoding


# Definieren Sie erneut abhängige und unabhängige Variablen
X_train = data[['attr_index_norm', "dist", "metro_dist", "realSum_Normalized", "cleanliness_rating"]]
y_train = data['guest_satisfaction_overall']


X_test = data_test[[ 'attr_index_norm', "dist", "metro_dist", "realSum_Normalized", "cleanliness_rating"]]
y_test = data_test['guest_satisfaction_overall']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
for column in X_train.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[column], y_train, alpha=0.5)
    plt.xlabel(column)
    plt.ylabel("guest_satisfaction_overall")
    plt.title(f"guest_satisfaction_overall vs. {column}")
    plt.grid(True)
    plt.show()

#Erstellen Sie eine Korrelationsmatrix
correlation_matrix = data.corr()

plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
plt.title("Korrelationsmatrix")
plt.show()
