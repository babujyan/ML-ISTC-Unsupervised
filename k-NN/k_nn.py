import numpy as np
import heapq

class K_NN:

    def __init__(self, k):
        """
        :param k: number of nearest neighbours
        """

        self.k = int(k)
        self.data = None


    def fit(self, data):
        """
        :param data: 3D array, where data[i, j] is i-th classes j-th point (vector: D dimenstions)
        """

        self.data = data


    def predict(self, data):
        """
        :param data: 2D array of floats N points each D dimensions
        :return: array of integers
        """
<<<<<<< HEAD

=======
>>>>>>> 7849db380755f8e4c745bb74da70fdfe4222b8a6
        data = np.array(data)
        shp = data.shape
        if len(data.shape) == 1:
            data = data.reshape([1] + list(data.shape))
<<<<<<< HEAD

        distance = {}
        s = []
        for datapoint in data:
            for i in range(self.data.shape[0]):

                for data1 in self.data[i]:
                    s.append([K_NN.euclideanDistance(data1,datapoint),i])


            s = sorted(s)
            a = np.array(s[0:self.k])
            unique , counts =np.unique(a[:,1],return_counts =True)
            max = np.argmax(counts)
            prediction = unique[max]
        return prediction.reshape(shp[:-1])

    def euclideanDistance(data1, data2):
        return np.sum((data1-data2)**2)**0.5
=======
        # TODO: predict
        prediction = np.array([0])
        return prediction.reshape(shp[:-1])
>>>>>>> 7849db380755f8e4c745bb74da70fdfe4222b8a6
