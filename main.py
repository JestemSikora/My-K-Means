# Importing libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from kmeans_model import My_Kmeans


# Loading data from sklearn load_iris 
iris = load_iris()
X, y = iris.data, iris.target

# k -> Number of Clusters
k = 3

# Importing my implementation
# of K-Means Algorithm
model_1 = My_Kmeans()

# Using model
model_1.fit(X, 3)



 
