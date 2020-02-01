from datetime import datetime as dt
import numpy as np
import shutil, os, subprocess, h5py, scipy.stats
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans

def wishartrand(nu, phi):
    dim = phi.shape[0]
    chol = np.linalg.cholesky(phi)
    foo = np.zeros((dim,dim))

    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i,j] = np.sqrt(scipy.stats.chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = np.random.normal(0,1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

def invwishartrand(rep, nu, phi):
    rv = []
    for r in range(rep):
        rv.append(np.linalg.inv(wishartrand(nu, np.linalg.inv(phi))))
    return np.array(rv)

class SimData:
    def __init__(self, outdir, seed, K, N, M, domain, proc):
        self.outdir = outdir
        self.seed = seed
        self.domain = domain
        self.proc = proc

        self.K = K
        self.N = N
        self.M = M

        self.beta = np.random.dirichlet(np.ones(self.K)*5)
        if domain == "unit":
            self.mu = np.random.normal(0.5, 0.2, (self.K, self.M))
            self.Sigma = invwishartrand(self.K, self.M, np.eye(self.M)/ \
                    (100000 if (proc >= 3) else 10000))
        elif domain == "positive" or domain == "integer":
            self.mu = np.random.normal(1000, 600, (self.K, self.M))
            self.Sigma = invwishartrand(self.K, self.M, np.eye(self.M))
        else:
            self.mu = np.random.normal(0, 1000, (self.K, self.M))
            self.Sigma = invwishartrand(self.K, self.M, np.eye(self.M))

    def transform(self, x): #inverse g
        if self.domain == "real":
            return x
        elif self.domain == "positive":
            x[x<100] = np.log(np.exp(np.maximum(x[x<100], 1e-6)) - 1)
            return x
        elif self.domain == "unit":
            b = 1e-6
            return np.log(((b + (1.-2*b)) / x) - 1) / -10 + 0.5
        elif self.domain == "integer":
            x[x<100] = np.log(np.exp(np.maximum(x[x<100], 1e-6)) - 1)
            return x
        else:
            raise Exception("Domain %s not allowed" % self.domain)

    def itransform(self, x): # g
        if self.domain == "real":
            return x
        elif self.domain == "positive":
            x[x<100] = np.log(np.exp(x[x<100]) + 1) # softplus
            return x
        elif self.domain == "unit":
            b = 1e-6
            return b+ (1.-2*b) / (1 + np.exp(-10*(x-0.5))) # sigmoid
        elif self.domain == "integer":
            x[x<100] = np.log(np.exp(x[x<100]) + 1) # softplus
            return x
        else:
            raise Exception("Domain %s not allowed" % self.domain)

    def draw(self, mu, sigma):
        if self.domain == "real":
            val = np.random.normal(self.itransform(mu), sigma)
        elif self.domain == "positive":
            mu = self.itransform(mu)
            a = (mu/sigma)**2
            b = sigma**2 / np.maximum(mu, 1e-6)
            val = np.random.gamma(a, b)
        elif self.domain == "unit":
            mu = self.itransform(mu)
            s = sigma * 1e-6 / max(sigma)
            a = ((1.0-mu)/(s**2) - 1.0/mu) * (mu**2)
            b = a * (1/mu - 1)
            val = np.random.beta(a, b)
        elif self.domain == "integer":
            val = np.random.poisson(self.itransform(mu))
        else:
            raise Exception("Domain %s not allowed" % self.domain)

        return val

    def save_settings(self, message):
        f = open(os.path.join(self.outdir, 'settings.dat'), 'w+')

        f.write("%s\n" % dt.now())
        f.write("%s\n\n" % message)

        f.write("random seed:\t%d\n\n" % self.seed)
        f.write("generation procedure:\t%d\n\n" % self.proc)
        f.write("domain of observations:\t%s\n\n" % self.domain)

        f.write("number of latent components K:\t%d\n" % self.K)
        f.write("number of observations N:\t%d\n" % self.N)
        f.write("number of features M:\t%d\n\n" % self.M)

        p = subprocess.Popen(['git','rev-parse', 'HEAD'], \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        f.write("\ncommit #%s" % out)

        p = subprocess.Popen(['git','diff', 'simulate.py'], \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        f.write("%s" % out)

        f.close()

    def save(self):
        f = open(os.path.join(self.outdir, 'data.hdf5'), 'w+')
        f.close()

        f = h5py.File(os.path.join(self.outdir, 'data.hdf5'), 'w')
        f.create_dataset("global_factor_concentration", data=self.beta)
        f.create_dataset("global_factor_features", data=self.mu) # in real space
        f.create_dataset("global_factor_feature_covariance", data=self.Sigma)
        f.create_dataset("local_factor_concentration", data=self.pi)
        f.create_dataset("particle_counts", data=self.P)
        f.create_dataset("observations", data=self.y)

        f.create_dataset("particle_features", data=self.x) # in the same space as observations
        f.create_dataset("particle_assignments", data=self.assignments) # to both N observations and K factors

        f.close()

    def simulate(self):
        pass

class SimDataProcedure1(SimData):
    def simulate(self):
        scales = np.random.gamma(4, 10, self.M)**-1
        self.Sigma = self.Sigma / 10
        self.pi = np.random.dirichlet(self.beta*0.3, self.N)
        self.P = np.random.poisson(100, self.N)
        self.assignments = np.zeros((sum(self.P), 2), dtype=int)
        self.x = np.zeros((sum(self.P), self.M))
        self.x_hat = np.zeros((self.N, self.K, self.M))
        i = 0
        for n in range(self.N):
            x_bar = np.zeros((self.K, self.M))
            x_hat = np.zeros((self.K, self.M))
            for k in range(self.K):
                x_bar[k] = np.random.multivariate_normal(self.mu[k], self.Sigma[k])
            for p in range(self.P[n]):
                z = list(np.random.multinomial(1, self.pi[n])).index(1)
                x = np.random.multivariate_normal(x_bar[z], self.Sigma[z]*1e-6)
                self.x[i] = x
                x_hat[z] += x
                self.assignments[i,0] = n
                self.assignments[i,1] = z
                i += 1
            self.x_hat[n] = x_hat
        self.y = self.draw(self.x_hat.sum(1) / self.P[:,np.newaxis], scales)
        self.x = self.itransform(self.x)

class SimDataProcedure2(SimData):
    def simulate(self):
        scales = np.random.gamma(1, 10, self.M)**-1
        self.pi = np.random.dirichlet(self.beta*0.3, self.N)
        self.P = np.random.poisson(100, self.N)
        self.assignments = np.zeros((sum(self.P), 2), dtype=int)
        self.x = np.zeros((sum(self.P), self.M))
        self.x_hat = np.zeros((self.N, self.K, self.M))
        i = 0
        for n in range(self.N):
            x_hat = np.zeros((self.K, self.M))
            for p in range(self.P[n]):
                z = list(np.random.multinomial(1, self.pi[n])).index(1)
                x = np.random.multivariate_normal(self.mu[z], self.Sigma[z])
                self.x[i] = x
                x_hat[z] += x
                self.assignments[i,0] = n
                self.assignments[i,1] = z
                i += 1
            self.x_hat[n] = x_hat
        self.y = self.draw(self.x_hat.sum(1) / self.P[:,np.newaxis], scales)
        self.x = self.itransform(self.x)

class SimDataProcedure3(SimData):
    def simulate(self):
        modes = np.random.poisson(5, self.K)
        local_mode_features = list()
        for k in range(self.K):
            local_mode_features.append(np.random.multivariate_normal(self.mu[k], self.Sigma[k], size=self.N))
        scales = np.random.gamma(20, 10, self.M)**-1
        self.pi = np.random.dirichlet(self.beta*0.2, self.N)
        self.P = np.random.poisson(100, self.N)
        self.assignments = np.zeros((sum(self.P), 2), dtype=int)
        self.x = np.zeros((sum(self.P), self.M))
        self.x_hat = np.zeros((self.N, self.K, self.M))
        i = 0
        for n in range(self.N):
            x_hat = np.zeros((self.K, self.M))
            gamma = []
            for p in range(self.P[n]):
                z = list(np.random.multinomial(1, self.pi[n])).index(1)
                x = self.draw(local_mode_features[z][n], scales)
                self.x[i] = x
                x_hat[z] += x
                self.assignments[i,0] = n
                self.assignments[i,1] = z
                i += 1
            self.x_hat[n] = x_hat
        self.y = self.x_hat.sum(1) / self.P[:,np.newaxis]
        if self.domain == "integer":
            self.y = np.asarray(self.y, dtype=np.int32)

class SimDataProcedure4(SimData):
    def simulate(self):
        modes = np.random.poisson(5, self.K)
        while sum(modes==0) > 0:
            modes[modes == 0] = np.random.poisson(5, sum(modes==0))
        mode_features = list()
        for k in range(self.K):
            mode_features.append(np.random.multivariate_normal(self.mu[k], self.Sigma[k], size=modes[k]))
        scales = np.random.gamma(20, 10, self.M)**-1
        self.pi = np.random.dirichlet(self.beta*0.2, self.N)
        self.P = np.random.poisson(100, self.N)
        self.assignments = np.zeros((sum(self.P), 2), dtype=int)
        self.x = np.zeros((sum(self.P), self.M))
        self.x_hat = np.zeros((self.N, self.K, self.M))
        i = 0
        for n in range(self.N):
            x_hat = np.zeros((self.K, self.M))
            gamma = []
            for k in range(self.K):
                gamma.append(np.random.dirichlet(np.ones(modes[k])*0.1))
            for p in range(self.P[n]):
                z = list(np.random.multinomial(1, self.pi[n])).index(1)
                s = list(np.random.multinomial(1, gamma[z])).index(1)
                x = self.draw(mode_features[z][s], scales)
                self.x[i] = x
                x_hat[z] += x
                self.assignments[i,0] = n
                self.assignments[i,1] = z
                i += 1
            self.x_hat[n] = x_hat
        self.y = self.x_hat.sum(1) / self.P[:,np.newaxis]
        if self.domain == "integer":
            self.y = np.asarray(self.y, dtype=np.int32)

class SimDataProcedure5(SimData):
    def simulate(self):
        scales = np.random.gamma(1, 10, self.M)**-1
        all_P = np.random.poisson(100*self.N)
        self.x = np.zeros((all_P, self.M))
        self.assignments = np.zeros((all_P, 2), dtype=int)
        for i in range(all_P):
            z = list(np.random.multinomial(1, self.beta)).index(1)
            x = np.random.multivariate_normal(self.mu[z], self.Sigma[z])
            self.x[i] = x
            self.assignments[i,1] = z

        # cluster x into N groups
        fuzzy_kmeans = FuzzyKMeans(k=self.N, m=2)
        fuzzy_kmeans.fit(self.x)

        self.pi = np.zeros((self.K, self.N))
        self.P = np.zeros(self.N)
        self.x_hat = np.zeros((self.N, self.K, self.M))
        for i in range(all_P):
            n = list(np.random.multinomial(1, fuzzy_kmeans.fuzzy_labels_[i])).index(1)
            self.assignments[i,0] = n
            self.x_hat[n][self.assignments[i,1]] += self.x[i]
            self.P[n] += 1
            self.pi[self.assignments[i,1],n] += 1
        # normalize pi
        self.pi = (self.pi.T / self.pi.sum(1).T).T
        self.y = self.draw(self.x_hat.sum(1) / self.P[:,np.newaxis], scales)
        self.x = self.itransform(self.x)

if __name__ == '__main__':
    ## Start by parsing the arguments
    import argparse

    # general script description
    parser = argparse.ArgumentParser(description = \
        'This script simulates data where observations are a convolution'
        'of latent factors.')

    parser.add_argument('--out', dest='outdir', type=str, \
        help='output directory', required=True)
    parser.add_argument('--msg', dest='message', type=str, \
        default='', help='log message')

    parser.add_argument('--seed', dest='seed', type=int, \
        default=(dt.fromtimestamp(0) - dt.now()).microseconds, \
        help='random seed, default from time')

    parser.add_argument('--domain', dest='domain', type=str, \
        default='real', help='domain for observations, '
        'default real.  options: real, positive, unit, integer.'
        '(unit is [0,1] and integer is positive integers only)')

    parser.add_argument('--K', dest='K', type=int, \
        default=10, help='number of global latent components, default 10')
    parser.add_argument('--N', dest='N', type=int, \
        default=100, help='number of observations, default 100')
    parser.add_argument('--M', dest='M', type=int, \
        default=10, help='number of features, default 10')

    parser.add_argument('--proc', dest='proc', type=int, \
        default=1, help='simulation procedure to be used.'
        'options 1-5 as described in Chaney, et al.')

    # parse the arguments
    args = parser.parse_args()

    # seed random number generator
    np.random.seed(args.seed)

    print(args.seed, args.outdir, args.domain, args.proc)

    # create output dir (check if exists)
    if os.path.exists(args.outdir):
        print("Output directory %s already exists.  Removing it to have a clean output directory!" % args.outdir)
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    # create an object to simulate data
    if args.proc == 1:
        data = SimDataProcedure1(args.outdir, args.seed, args.K, args.N, args.M, args.domain, 1)
    elif args.proc == 2:
        data = SimDataProcedure2(args.outdir, args.seed, args.K, args.N, args.M, args.domain, 2)
    elif args.proc == 3:
        data = SimDataProcedure3(args.outdir, args.seed, args.K, args.N, args.M, args.domain, 3)
    elif args.proc == 4:
        data = SimDataProcedure4(args.outdir, args.seed, args.K, args.N, args.M, args.domain, 4)
    elif args.proc == 5:
        data = SimDataProcedure5(args.outdir, args.seed, args.K, args.N, args.M, args.domain, 5)

    # simulate data and save
    data.simulate()
    data.save_settings(args.message)
    data.save()
