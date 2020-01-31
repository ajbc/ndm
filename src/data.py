import h5py
import numpy as np

class BatchIter:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def __iter__(self):
        return self

    def next(self):
        if self.i < self.data.N:
            i = self.i
            self.i += 1
            return (i, self.data.obs[i])
        else:
            raise StopIteration()

class SampledIter:
    def __init__(self, data, M):
        self.data = data
        self.i = 0
        self.M = M

    def __iter__(self):
        return self

    def next(self):
        if self.i < self.M:
            i = np.random.randint(self.N)
            self.i += 1
            return (i, self.data.obs[i])
        else:
            raise StopIteration()


class Data:
    def __init__(self, observations):
        f = h5py.File(observations, 'r')
        self.obs = f['observations']
        self.N = self.obs.shape[0] # num observations
        self.M = self.obs.shape[1] # num features

        self.known_density = False
        if 'density' in f:
            self.known_density = True
            self.density = f['density']

        #TODO: remove this cheating (for piecewise checks)
        '''if "global_factor_concentration" in f:
            self.beta = f["global_factor_concentration"]
            self.mu = f["global_factor_features"]
            self.sigma = f["global_factor_feature_covariance"]
            self.pi = f["local_factor_concentration"]
            self.x = f["local_factor_features"]
            self.K = self.beta.shape[0]
            self.eta = f["global_features"]'''

    def batch_iter(self):
        return BatchIter(self)

    def sampled_iter(self, M):
        return SampledIter(self, M)
