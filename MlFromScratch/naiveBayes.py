import numpy as np

class Bayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        #calculate mean vairance and prior for each class
        #print(self._classes)
        self._mean = np.zeros((n_classes, n_features), dtype = np.float64)
        self._var = np.zeros((n_classes, n_features), dtype = np.float64)
        self._priors = np.zeros(n_classes, dtype = np.float64)
        
        print(self._classes)
        print(self._mean)
        print(self._var)
        print(self._priors)
        print()

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis = 0)
            self._var[idx, :] = X_c.var(axis = 0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)
        
        print(self._classes)
        print(self._mean)
        print(self._var)
        print(self._priors)
            
        def predict(self, X):
            predicted = [self._predicted(x) for x in X]
            return np.array(predicted)

        def _predicted(self, x):
            posteriors = []

            #calculate the posteriors proballity for each class
            for idx, c in enumerate(self._classes):
                prior = np.log(self._priors[idx])
                posterior = np.sum(np.log(self._pdf(idx, x)))
                posterior = prior + posterior
                posteriors.append(posterior)

            #return class with the highest posteriors proballity
            #argmax return the index
            return self._classes[np.argmax(posteriors)]
        
        #proballity distribution function
        def _pdf(self, class_id, x):
            mean = self._mean[class_id]
            var = self._var[class_id]
            numerator = np.exp(-(x - mean)** 2 / (2 * var))
            denominator = np.sqrt(2 * np.pi * var)
            return numerator / denominator




X = [[1.0, 1.1], [2.2, 3.2], [3.2, 2.5], [1.3, 4.2], [2.4, 1.1]]
y = [0, 1, 0, 0, 2]

b = Bayes()
b.fit(np.asarray(X), np.asarray(y))
b.predict(np.array([[1.1, 2.2], [2.1, 3.2]])) 











