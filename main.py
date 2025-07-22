import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from kmeans_model import My_Kmeans
import random

k = 3
# Loading data
iris = load_iris()
X, y = iris.data, iris.target

#print(f'Shapes of X and y: {X.shape, y.shape}')
#print(y)
#print(f'max: {X.max()}, min: {X.min()}')


model_1 = My_Kmeans()
model_1.random_location_pick(X,3)
model_1.fit(X, 3)



 
