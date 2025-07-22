
import pandas as pd
import numpy as np
import random


class My_Kmeans:
    def __init__(self):
        self.k = {}
        self.position = np.array([])

    def random_location_pick(self, X, k):
        ''' Zrobić z odchylenia takie punkty, aby pokazywały się w granicach X'''
        mean_ = np.mean(X)
        std_ = np.std(X)

        self.position = np.random.normal(loc=mean_, scale=std_, size=(k, X.shape[1]))
        self.position = np.round(self.position, 2)


    def mean_function(self, X, k, max_attempts=10):
        attempt = 0

        while attempt < max_attempts:
            restart_needed = False

            for i in range(k):
                cluster_values = self.k[f'{i}th']
                s_j = len(cluster_values)

                if s_j == 0:
                    self.position[i] = X[np.random.choice(len(X))]
                    self.claster_function(X, k, True, i)
                    restart_needed = True
                    break

                total = sum(cluster_values)
                self.position[i] = np.round(total / s_j, 2)

            if not restart_needed:
                break
            attempt += 1

        if attempt == max_attempts:
            raise RuntimeError("Too many attempts to fix empty clusters")

            
    def claster_function(self, X, k, Null_Claster, null_idx):

        if Null_Claster is True and null_idx is not None:
            self.k = {f'{i}th': [] for i in range(k)}
            distance = np.empty((X.shape[0], k))

            for i in range(X.shape[0]):
                distance[i] = np.linalg.norm(X[i] - self.position[null_idx])

                claster_num = np.argmin(distance[i])
                self.k[f'{claster_num}th'].append(X[i])

        else:
            self.k = {f'{i}th': [] for i in range(k)}
            distance = np.empty((X.shape[0], k))

            for i in range(X.shape[0]):
                for j in range(k):
                    distance[i, j] = np.linalg.norm(X[i] - self.position[j])

                claster_num = np.argmin(distance[i])
                self.k[f'{claster_num}th'].append(X[i])





    def fit(self, X, k):
        for i in range(101):
            if not self.position.any():
                self.random_location_pick(X, k)

            self.claster_function(X, k, False, None)
            self.mean_function(X, k)
            if i % 10 == 0:
                print(f'Clustering for {i}th iteration: {[f"{k}: {len(v)}" for k, v in self.k.items()]}')
                print(f'New position after {i}th iteration: {self.position}')
                print("---------------------------------------------")





