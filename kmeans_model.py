import pandas as pd
import numpy as np
import random

class My_Kmeans:
    def __init__(self):
        self.k = {}
        self.position = []

    def random_location_pick(self, X, k):
        ''' Zrobić z odchylenia takie punkty, aby pokazywały się w granicach X'''
        mean_ = np.mean(X)
        std_ = np.std(X)

        self.position = np.random.normal(loc=mean_, scale=std_, size=(3, X.shape[1]))
        self.position = np.round(self.position, 2)


    def mean_function(self, X, k):
        for i in range(k):
            cluster_values = self.k[f'{i}th'].values()
            total = sum(cluster_values)

            s_j = len(self.k[f'{i}th'].values())

            self.position = 1 / (s_j) * total

            

    def claster_function(self, X, k):
        self.k = {f'{i}th': [] for i in range(k)}
        distance = np.empty((X.shape[0], k))

        for i in range(X.shape[0]):
            for j in range(k):
                distance[i, j] = np.linalg.norm(X[i] - self.position[j])

                claster_num = np.argmin(distance[i])
                self.k[f'{claster_num}th'].append(X[i])


    def fit(self, X, k):
        for i in range(10):
            if not self.position:
                self.random_location_pick(X, k)

            self.claster_function(X, k)
            self.mean_function(X, k)





