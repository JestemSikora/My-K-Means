# My-K-Means
K Means Clustering - Iris Dataset

Projekt implementuje algorytm K Means od podstaw w Pythonie na zbiorze danych Iris.

Pliki:
- main.py – uruchamia algorytm K Means na danych Iris.
- kmeans_model.py – zawiera klasę My_Kmeans z własną implementacją K Means.

Wymagania:
- Python 3.x
- NumPy
- Pandas
- scikit-learn

Uruchomienie:
1. Zainstaluj wymagane biblioteki:
   pip install numpy pandas scikit-learn
2. Uruchom plik main.py:
   python main.py

Opis działania:
- Dane Iris są ładowane z sklearn.datasets.
- Algorytm K Means dzieli dane na 3 klastry.
- Wyniki klastrów są dostępne w atrybucie model_1.k.

Przykład użycia:
from kmeans_model import My_Kmeans
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

model = My_Kmeans()
model.fit(X, 3)

for key, cluster in model.k.items():
    print(f"{key}: {len(cluster)} punktów")

Kontakt:
