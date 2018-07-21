import numpy as np


class PCA:
   def __init__(self, k):
       self.k = k
       self.location_ = None
       self.matrix_ = None

   def fit(self, data):
       """
       finds best params for X = Mu + A * Lambda
       :param data: data of shape (number of samples, number of features)
       HINT! use SVD
       """
       #data = data / np.linalg.norm(data, axis=0)
       mean_ = np.mean(data, axis=0)
       data -= mean_
       #         cov_matix = np.cov(data.T.dot(data))
       #         w , v  = np.linalg.eig(cov_matix)

       u, s, v = np.linalg.svd(data, full_matrices=False)
       self.matrix_ = v[:self.k]

   def transform(self, data):
       """
       for given data returns Lambdas
       x_i = mu + A dot lambda_i
       where mu is location_, A is matrix_ and lambdas are projection of x_i
       on linear space from A's rows as basis
       :param data: data of shape (number of samples, number of features)
       """
       # Lemma: x is vector and A dot A.T == I, then x's coordinates in Linear Space(A's rows as basis)
       # is A dot x

       return np.dot(self.matrix_, data.T).T

   def return_components(self):
       return self.matrix_