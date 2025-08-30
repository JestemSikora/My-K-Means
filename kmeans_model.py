# Importing libraries
import pandas as pd
import numpy as np
import random

# Making class to implement K-Means Logic
class My_Kmeans:
    def __init__(self):
        self.k = {}
        self.position = np.array([])



    def random_location_pick(self, X, k):
        '''Picking randomly first positions of k Clusters; dimension [k, X.shape[1]]'''
        idx = np.random.choice(len(X), size=k, replace=False)
        self.position = np.round(X[idx], 2)



    def mean_function(self, X, k, max_attempts=10):
        '''Calculating mean of each Cluster and updating positions of new Clusters'''
        attempt = 0

        # Checking for to many Erorrs
        while attempt < max_attempts:
            restart_needed = False

            # For each Cluster, calculating new mean from sum and length
            for i in range(k):
                cluster_values = self.k[f'{i}th']
                s_j = len(cluster_values)

                # Preventing dividing by 0
                # if 0 occurres, picking new position for this Centroid
                if s_j == 0:
                    self.position[i] = X[np.random.choice(len(X))]
                    self.claster_function(X, k, True, i) 
                    restart_needed = True
                    break # Reseting mean_function

                total = np.sum(cluster_values, axis= 0)
                self.position[i] = np.round(total / s_j, 2)

            if not restart_needed:
                break
            attempt += 1

        # Error Information
        if attempt == max_attempts:
            raise RuntimeError("Too many attempts to fix empty clusters")



    def claster_function(self, X, k, Null_Claster, null_idx):
        '''Clastering data'''

        # Checking if calling this function is caused by
        # length of a Cluster being equeal to 0
        if Null_Claster is True and null_idx is not None:
            self.k = {f'{i}th': [] for i in range(k)}
            distance = np.empty((X.shape[0], k))

            for i in range(X.shape[0]):
                distance[i] = np.linalg.norm(X[i] - self.position[null_idx])

                claster_num = np.argmin(distance[i])
                self.k[f'{claster_num}th'].append(X[i])

        else:
            self.k = {f'{i}th': [] for i in range(k)} # Making dict. for Clusters
            distance = np.empty((X.shape[0], k))

            for i in range(X.shape[0]):
                for j in range(k):
                    distance[i, j] = np.linalg.norm(X[i] - self.position[j]) # Calculating distance by substracting each 
                                                                                # row of self.position of Centroids

                claster_num = np.argmin(distance[i]) # Finding the closest Centroid
                self.k[f'{claster_num}th'].append(X[i]) # Saving to Cluster



    def fit(self, X, k):
        '''Fitting data to Algorithm'''
        for i in range(21):
            
            # First iteration -> randomly selecting positions of Centroids
            if not self.position.any():
                self.random_location_pick(X, k)

            self.claster_function(X, k, False, None)
            self.mean_function(X, k)

            if i % 5 == 0:
                print(f'Clustering for {i}th iteration: {[f"{k}: {len(v)}" for k, v in self.k.items()]}')
                print(f'New position of centroids after {i}th iteration: {self.position}')
                print("---------------------------------------------")






