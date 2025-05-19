import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pickle

# Cargar dataset
data = pd.read_csv("carrisk-future.csv")  # Asegúrate que el CSV esté en la misma carpeta

# Preprocesamiento
le_risk = LabelEncoder()
data['risk_encoded'] = le_risk.fit_transform(data['risk'])  # Ajusta 'risk' si tu columna se llama distinto

# Variables predictoras y etiqueta
X = data.drop(columns=['risk', 'risk_encoded'])
y = data['risk_encoded']

# One-hot encoding para variables categóricas
X = pd.get_dummies(X, drop_first=False)

# Escalar variables numéricas (ajusta si tienes más)
numericas = ['age']
X[numericas] = MinMaxScaler().fit_transform(X[numericas])

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelos
model_tree = DecisionTreeClassifier().fit(X_train, y_train)
model_knn = KNeighborsClassifier().fit(X_train, y_train)
model_nn = MLPClassifier(max_iter=1000).fit(X_train, y_train)

# Guardar modelos y objetos
variables = X.columns.tolist()
scaler = MinMaxScaler().fit(X[numericas])

with open("modelo-clas-tree-knn-nn.pkl", "wb") as f:
    pickle.dump((model_tree, model_knn, model_nn, le_risk, variables, scaler), f)

print("Modelos entrenados y guardados correctamente.")
