#KNN for K neighbours
#check for K nearest neighbours and then decides according to this
#To calculate the distance we use the euclead distance
# d = sqrt((x1 - x2) * (x1 - x2) + (y1 - y1) * (y1 - y2))
#for N dimension sqrt(sum(i = {0..N}) (x1 - xi) * (x1 - xi))


from collections import Counter 
import numpy as np

class KNN:
    def __init__(self,K = 3):
        self.K = K
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_value = [self._predict(x) for x in X]
        return np.array(predicted_value)

    def euclidean(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _predict(self, x):
        distances = [self.euclidean(x, x_train) for x_train in self.X_train]
        #sort by distance and return indices of first k neighbours
        k_idx = np.argsort(distances)[:self.K]
        k_neighbour_labels  = [self.y_train[i] for i in k_idx]
        #return the most common label
        most_common = Counter(k_neighbour_labels).most_common()
        return most_common[0][0]











