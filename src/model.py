import h5py, os, time, sys
import numpy as np
from scipy.special import gammaln, digamma, multigammaln
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from multiprocessing.pool import Pool

#import warnings
np.seterr(all='raise')
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=140)

def softplus(x):
    trunc = np.log(np.finfo(np.float64).max)
    min_trunc = softplus_inverse(1e-6)
    if np.isscalar(x):
        if x > trunc:
            return x
        else:
            try:
                v = np.log(np.exp(x) + 1)
            except:
                v = 0
            return v

            return np.log(np.exp(x) + 1)
    trunc_x = np.array(x, dtype=np.float64)
    trunc_x[trunc_x > trunc] = trunc
    trunc_x[trunc_x < min_trunc] = min_trunc
    try:
        val = np.log(np.exp(trunc_x) + 1)
    except:
        print(trunc)
        print(trunc_x)
    val[trunc_x==trunc] = x[trunc_x==trunc]
    val[trunc_x==min_trunc] = x[trunc_x==min_trunc]
    return val

def softplus_inverse(x):
    hi_trunc = np.log(np.finfo(np.float32).max)
    lo_trunc = 1e-10
    if np.isscalar(x):
        if x > hi_trunc:
            return x
        elif x < lo_trunc:
            return np.log(np.exp(lo_trunc) - 1)
        else:
            return np.log(np.exp(x) - 1)
    trunc_x = np.array(x, dtype=np.float64)
    trunc_x[trunc_x > hi_trunc] = hi_trunc
    trunc_x[trunc_x < lo_trunc] = lo_trunc
    val = np.log(np.exp(trunc_x) - 1)
    val[trunc_x==hi_trunc] = x[trunc_x==hi_trunc]
    return val

def softplus_derivative(x):
    trunc = np.log(np.finfo(np.float64).max)
    if np.isscalar(x):
        if x > trunc:
            return 1.0
        else:
            return np.float64(np.exp(x) / (1. + np.exp(x)))
    rv = np.ones(x.shape)
    rv[x <= trunc] = np.float64(np.exp(x[x <= trunc]) / (1. + np.exp(x[x <= trunc])))
    return rv

def covar(a, b):
    # (s, M)
    v = (a - sum(a)/a.shape[0]) * (b - sum(b)/b.shape[0])
    return sum(v) / v.shape[0]

def var(a):
    rv = covar(a, a)
    return np.maximum(1e-300, rv)

def logfactorial(x):
    if np.isscalar(x):
        return np.log(np.arange(1,x+1)).sum()
    rv = np.zeros(x.shape)
    if len(rv.shape) == 1:
        for i in range(len(x)):
            rv[i] = np.log(np.arange(1,x[i]+1)).sum()
    else:
        for i in range(rv.shape[0]):
            for j in range(rv.shape[1]):
                rv[i,j] = np.log(np.arange(1,x[i,j]+1)).sum()
    return rv

# the following two functions from https://gist.github.com/jfrelinger/2638485
def invwishartrand(nu, phi):
    return np.linalg.inv(wishartrand(nu, np.linalg.inv(phi)))

def wishartrand(nu, phi):
    dim = phi.shape[0]
    chol = np.linalg.cholesky(phi)
    foo = np.zeros((dim,dim))

    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = np.random.normal(0,1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

def logGamma(x, mu, alpha):
    x = np.maximum(x, 1e-6)
    shape = mu / alpha
    scale = alpha
    return ((shape-1)*np.log(x) - (x/scale) - \
            gammaln(shape) - shape*np.log(scale))

def logiGamma(x, alpha, beta):
    return alpha * np.log(beta) - gammaln(alpha) - \
            (alpha-1) * np.log(x) - beta / x

def logNormal(x, loc, var):
    diff = x - loc
    thresh = np.sqrt(np.finfo(np.float64).max / 2)
    diff[diff > thresh] =  thresh
    return -0.5 * (np.log(2 * np.pi * var) + \
            diff ** 2 / var)

def logMVNormal(x, mu, Sigma, iSigma, detSigma, scale):
    try:
        return -0.5 * (detSigma - Sigma.shape[0] * np.log(scale) + \
                np.matmul(np.matmul(np.transpose(x - mu), \
                iSigma * scale), x - mu))
    except:
        print(Sigma)
        raise

def logiWishart(x, df, scale):
    d = scale.shape[0]
    sign, logdetScale = np.linalg.slogdet(scale)
    sign, logdetX = np.linalg.slogdet(x)

    rv = (df * 0.5) * logdetScale - \
            (df * d * 0.5) * np.log(2) - \
            multigammaln(df * 0.5, d) - \
            ((df + d + 1) * 0.5) * logdetX - \
            0.5 * np.matmul(scale, np.linalg.inv(x)).diagonal().sum()

    return rv

def logBetaShape(x, alpha, beta):
    x = np.minimum(np.maximum(alpha, 1e-10), 1-1e-10)
    alpha = np.maximum(alpha, 1e-10)
    beta = np.maximum(beta, 1e-10)
    return (alpha - 1.0) * np.log(x) + \
            (beta - 1.0) * np.log(1.0 - x) - \
            gammaln(alpha) - gammaln(beta) + \
            gammaln(alpha + beta)

def logDPBeta(x, alpha0):
    # unnormalize
    ubeta = np.zeros(x.shape)
    remainder = 1.0
    for i in range(len(ubeta)-1):
        ubeta[i] = x[i] / remainder
        remainder *= np.maximum((1.0 - x[i]), 1e-100)
    ubeta[-1] = remainder

    # compute log prob
    rv = 0
    for i in range(len(ubeta)-1):
        rv += logBetaShape(x[i], 1.0, alpha0)
    return rv

def logDPGamma(x, alpha):
    rv = ((alpha[:x.shape[1]]-1) * np.log(x) - x).sum(1)
    rv -= gammaln(alpha).sum()
    return rv

def logDirichlet(x, alpha):
    return ((alpha - 1) * np.log(x)).sum() + \
            gammaln(alpha.sum()) - \
            gammaln(alpha).sum()

def loggDirichlet(x, alpha):
    return np.log(x) + digamma(np.sum(alpha)) - digamma(alpha)

def loggmGamma(x, mu, alpha):
    return (- (alpha / mu) + ((alpha * x) / mu**2))

def loggaGamma(x, mu, alpha):
    return (np.log(alpha) + 1. - np.log(mu) - digamma(alpha) + np.log(x) - (x / mu))

def loggaiGamma(x, alpha, beta):
    return np.log(beta) - digamma(alpha) - np.log(x)

def loggbiGamma(x, alpha, beta):
    return alpha / beta - 1. / x

def logPoisson(x, rate):
    if np.isscalar(rate):
        if rate < 1e-100:
            rate = 1e-100
    else:
        rate[rate < 1e-100] = 1e-100
    return x * np.log(rate) - rate - logfactorial(x)

def loggPoisson(x, rate):
    return x / rate - 1.0

def logBeta(x, mean, var):
    mean = np.minimum(np.maximum(mean, 1e-6), 1.0-1e-6)
    alpha = (((1 - mean) / var) - 1. / mean) * (mean**2)
    beta = alpha * (1./mean - 1)
    x[x < 1e-10] = 1e-10
    x[x > 1. - 1e-6] = 1. - 1e-6
    return (alpha - 1) * np.log(x) + (beta - 1) * np.log(1 - x) + \
            gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)

def log_gloc_Normal(x, loc, scl):
    return (x - loc) / (scl**2)

def log_gscl_Normal(x, loc, scl):
    return (-1 / scl) + ((x - loc) ** 2) * (scl ** -3)

def loggMVNormal(x, mu, Sigma):
    return np.matmul(np.linalg.inv(Sigma), \
            (x - mu))

def sampleDirichlet(alpha):
    draws = 0
    success = False
    while not success and draws < 5:
        try:
            s = np.random.dirichlet(alpha)
            if np.isnan(s.sum()) or np.isinf(s.sum()):
                success = False
            else:
                success = True
        except:
            draws += 1
            # if the alpha values are all too low, this messes up sampling
            if alpha.sum() < 1e-2:
                alpha = alpha * 10

    return np.maximum(s, 1e-8)

def sampleGamma(mu, alpha):
    return np.random.gamma(mu / alpha, alpha)

def sampleiGamma(alpha, beta):
    return 1. / np.random.gamma(alpha, beta)

class Parameter:
    def mean(self):
        pass

    def logq(self):
        pass

class DirichletParameter(Parameter):
    def set_alpha(self, val):
        self.alpha = np.clip(val, 1e-6, 1e6)

    def mean(self):
        if len(self.alpha.shape) == 1:
            return self.alpha / self.alpha.sum()
        return (self.alpha.T/self.alpha.sum(1)).T

    def logq(self):
        return logDirichlet(self.mean(), self.alpha).sum()

class GammaParameter(Parameter):
    def set_mu(self, val):
        self.mu = np.clip(val, 1e-6, 1e6)

    def set_alpha(self, val):
        self.alpha = np.clip(val, 1e-6, 1e6)

    def mean(self):
        return self.mu

    def logq(self):
        return logGamma(self.mean(), self.mu, self.alphas).sum()

class iWishartParameter(Parameter):
    def __init__(self):
        self.new_params = True

    def vals(self):
        if self.new_params:
            self.mean()
        return self.saved_mean, self.saved_invmean, self.saved_det

    def mean(self):
        if self.new_params:
            rv = np.zeros(self.scale.shape)
            self.saved_invmean = np.zeros(self.scale.shape)
            self.saved_det = np.zeros(self.df.shape[0])
            for k in range(self.df.shape[0]):
                if self.df[k] > self.scale.shape[1] + 1:
                    rv[k] = self.scale[k] / (self.df[k] - self.scale.shape[1] - 1)
                else:
                    #print k, "sampling", self.df[k], self.scale.shape[1]
                    # no analytic solution, need to sample; this shouldn't happen frequently
                    samples = np.array([invwishartrand(self.df[k], self.scale[k]) for i in range(10000)])
                    rv[k] = np.mean(samples, 0)
                self.saved_invmean[k] = np.linalg.inv(rv[k])
                sign, logdet = np.linalg.slogdet(2 * np.pi * rv[k])
                self.saved_det[k] = logdet
            self.new_params = False
            self.saved_mean = rv
        return self.saved_mean

    def inv_mean(self):
        if self.new_params:
            self.mean()
        return self.invmean

    def set_scale(self, val):
        self.scale = val
        self.new_params = True

    def set_df(self, val):
        self.df = val
        self.new_params = True

    def logq(self):
        rv = 0
        mean = self.mean()
        for k in range(self.df.shape[0]):
            rv += logiWishart(mean[k], self.df[k], self.scale[k])
        return rv

class NormalParameter(Parameter):
    def __init__(self, minval, maxval):
        self.min = minval
        self.max = maxval

    def set_loc(self, val):
        self.loc = np.clip(val, self.min, self.max)

    def set_var(self, val):
        self.var = np.maximum(val, 1e-6)

    def mean(self):
        return self.loc

    def logq(self):
        return logNormal(self.mean(), self.loc, self.var).sum()

class PoissonParameter(Parameter):
    def __init__(self):
        self.rate = 1.0

    def set_rate(self, rate):
        #self.rate = np.clip(np.softplus(rate), 1e-6, 1e6)
        self.rate = np.clip(rate, 1e-6, 1e6)

    def mean(self):
        return self.rate

    def logq(self):
        return logPoisson(self.mean(), self.rate).sum()

def loglikelihood(g_link, f_dist, obs, means, eta):
    if g_link == "exp":
        means = np.exp(means)
    elif g_link == "softplus":
        means = softplus(means)
    elif g_link == "sigmoid":
        means[means < -100] = -100
        means[means > 100] = 100
        means = 1. / (1 + np.exp(-means))
    elif g_link == "expinverse":
        means = np.exp(means) ** -1

    if f_dist == "normal":
        return logNormal(obs, means, eta)
    elif f_dist == "log_normal":
        return np.exp(logNormal(obs, means, eta))
    elif f_dist == "gamma":
        return logGamma(obs, means, means**2 / eta)
    elif f_dist == "exponential":
        return logExponential(obs, means)
    elif f_dist == "poisson":
        return logPoisson(obs, means)
    elif f_dist == "beta":
        return logBeta(np.array(obs), means, eta)

# Parallelization helper functions

def get_local_count_pqg(S, count, rho, K, x, mu, Sigma, iSigma, detSigma, pi):
    # sample P
    s = np.asfarray(np.maximum(np.random.poisson(count, S), 1))

    p = logPoisson(s, rho)
    q = logPoisson(s, count)
    g = loggPoisson(s, count) * softplus_derivative(softplus_inverse(count))
    for i in range(S):
        for k in range(K):
            p[i] += logMVNormal(x[k], mu[k], \
                    Sigma[k], iSigma[k], detSigma[k], (s[i] * pi[k]))

    return p, q, g

def get_local_factors_bb(K, S, M, pi, x_loc, x_scl, mu, Sigma, iSigma, detSigma, count, obs, \
        MS_loc, MS_scl, rho_loc, rho_scl, g, f, eta):
    for k in sorted(range(K), key=lambda x: np.random.random()):

        p = np.zeros((S, M))
        q = np.zeros((S, M))
        g_loc = np.zeros((S, M))
        g_scl = np.zeros((S, M))

        others = np.matmul(pi, x_loc) - (x_loc[k] * pi[k])

        for i in range(S):
            # sample x
            s = np.random.normal(x_loc[k], x_scl[k])

            p[i] = logMVNormal(s, mu[k], \
                    Sigma[k], iSigma[k], detSigma[k], (count * pi[k])) + \
                    loglikelihood(g, f, obs, s*pi[k] + others, eta)

            q[i] = logNormal(s, x_loc[k], x_scl[k]**2)
            g_loc[i] = log_gloc_Normal(s, x_loc[k], x_scl[k])
            g_scl[i] = log_gscl_Normal(s, x_loc[k], x_scl[k]) * softplus_derivative(softplus_inverse(x_scl[k]))

        # RMSprop: keep running average of gradient magnitudes
        # (the gradient will be divided by sqrt of this later)
        if MS_loc[k].all() == 0:
            MS_loc[k] = (g_loc**2).sum(0)
            MS_scl[k] = (g_scl**2).sum(0)
        else:
            MS_loc[k] = 0.9 * MS_loc[k] + \
                    0.1 * (g_loc**2).sum(0)
            MS_scl[k] = 0.9 * MS_scl[k] + \
                    0.1 * (g_scl**2).sum(0)

        cv_loc = covar(g_loc * (p - q), g_loc) / var(g_loc)
        cv_scl = covar(g_scl * (p - q), g_scl) / var(g_scl)
        x_loc[k] += rho_loc * (1. / S) * \
                ((g_loc/np.sqrt(MS_loc[k])) * (p - q - cv_loc)).sum(0)
        x_scl[k] = softplus(softplus_inverse(x_scl[k]) + rho_scl * (1. / S) * \
                ((g_scl/np.sqrt(MS_scl[k])) * (p - q - cv_scl)).sum(0))

    return x_loc, x_scl, MS_loc, MS_scl

class Model:
    def __init__(self, settings, data):
        self.settings = settings
        self.data = data
        self.last_label = ''

        # Set up processes pool if needed
        if self.settings.cores > 1:
            self.pool = Pool(processes=self.settings.cores)

        # seed random number generator
        np.random.seed(settings.seed)

        # precompute this for mu and Sigma estimates
        self.sigma_prior = np.identity(data.M) * \
                np.var(data.obs, 0)
        self.inv_sigma_prior = np.linalg.inv(self.sigma_prior)
        self.mu_sigma_prior = np.var(data.obs, 0)
        self.mu_prior = np.mean(data.obs, 0)
        self.sigma_mu_prior = np.matmul(self.inv_sigma_prior, \
                self.mu_prior) * np.identity(data.M)

        # precompute transformed observations
        if self.settings.g_link == "exp":
            self.data.transformed_obs = np.log(self.data.obs)
        elif self.settings.g_link == "softplus":
            self.data.transformed_obs = softplus_inverse(self.data.obs)
        elif self.settings.g_link == "sigmoid":
            trunc_obs = np.array(self.data.obs, dtype=np.float64)
            trunc_obs[trunc_obs < 1e-100] = 1e-100
            trunc_obs[trunc_obs > 1. - 1e-10] = 1. - 1e-10
            print(np.min(trunc_obs), np.max(trunc_obs))
            self.data.transformed_obs = -np.log(1./trunc_obs - 1)
        elif self.settings.g_link == "expinverse":
            self.data.transformed_obs = np.log(self.data.obs ** -1)
        else:
            self.data.transformed_obs = np.array(self.data.obs)

        # global feature variances, if needed
        self.eta = 1e-5 * np.var(self.data.obs, 0)

        # specification and defining q for inference
        self.adopt_submodel(self.Submodel(self, K=settings.K))
        # TODO: set/init K more intelligently (based on data?)

    def loglikelihood(self, obs, means):
        return loglikelihood(self.settings.g_link, \
                self.settings.f_dist, obs, means, self.eta)

    class Submodel:
        def __init__(self, model, K, merge=None, split=None):
            self.K = K
            self.model = model

            # Sanity check: cannot both split and merge
            if merge is not None and split is not None:
                raise ValueError("Cannot both split and merge.")
            if merge is not None:
                print("MERGING", self.K, merge)
            elif split is not None:
                print("SPLITTING", self.K, split)
            else:
                print("INIT w/o split or merge", self.K, split, merge)

            # set up initial model settings using kmeans
            if merge is None and split is None:
                # adding random noise to the observations avoids a
                # divide-by-zero error in FuzzyKMeans when data is discrete
                # (cluster centers can match up with data exactly)
                km_data = model.data.obs + \
                        np.var(model.data.obs, 0) * np.random.random(model.data.obs.shape) * 1e-6
                km = FuzzyKMeans(k=model.settings.K, m=2).fit(km_data)
                km.weights = np.sum(km.fuzzy_labels_, axis=0)
            if split is not None:
                # sample 1000 data points weighted by assignment to the split factor
                obs_idxs = np.random.choice(range(model.data.N), 1000, \
                        p=model.submodel.qpi.mean()[:,split] / np.sum(model.submodel.qpi.mean()[:,split]))
                km_data = np.array(model.data.obs)[obs_idxs] + \
                        np.var(model.data.obs, 0) * np.random.random((1000, model.data.M)) * 1e-6
                km = FuzzyKMeans(k=2, m=2).fit(km_data)
                km.weights = np.sum(km.fuzzy_labels_, axis=0)


            ### INFERENCE: define q / variational parameters
            ## factor proportions
            # global proportions
            self.qbeta = DirichletParameter()
            if merge is not None:
                init = model.submodel.qbeta.alpha
                init[merge[0]] += init[merge[1]]
                init = np.delete(init, merge[1])
                self.qbeta.set_alpha(init)
            elif split is not None:
                init = model.submodel.qbeta.alpha
                #rho = (model.iteration + 4) ** -0.5
                #init = np.insert(init, -1, rho * init[split])
                #init[split] = (1-rho) * init[split]
                prop = km.fuzzy_labels_.sum(0)
                init = np.insert(init, -1, init[split] * prop[1] / prop.sum())
                init[split] = (prop[0] / prop.sum()) * init[split]
                self.qbeta.set_alpha(init)
            else:
                alpha_init = km.fuzzy_labels_.sum(0)

                scale = np.mean((((alpha_init * (alpha_init.sum() - alpha_init)) / \
                        (np.var(km.fuzzy_labels_, 0) * alpha_init.sum()**2)) - 1) / \
                        alpha_init.sum())
                if model.settings.fix_K:
                    self.qbeta.set_alpha(alpha_init * scale)
                else:
                    self.qbeta.set_alpha(np.append(alpha_init * scale, model.settings.gbl_alpha * scale))

            # local proportions
            self.qpi = DirichletParameter()
            if merge is not None:
                init = model.submodel.qpi.alpha
                init[:,merge[0]] += init[:,merge[1]]
                init = np.delete(init, merge[1], axis=1)
                self.qpi.set_alpha(init)
            elif split is not None:
                init = model.submodel.qpi.alpha
                #init = np.insert(init, -1, rho * init[:,split], axis=1)
                #init[:,split] = (1-rho) * init[:,split]
                D = 1.0 / euclidean_distances(model.data.obs, km.cluster_centers_, squared=True)
                D /= np.sum(D, axis=1)[:, np.newaxis]
                init = np.insert(init, -1, D[:,1]/D.sum(1) * init[:,split], axis=1)
                init[:,split] = (D[:,0]/D.sum(1)) * init[:,split]
                self.qpi.set_alpha(init)
            else:
                if model.settings.fix_K:
                    self.qpi.set_alpha(np.maximum(0.001, km.fuzzy_labels_)*100)
                else:
                    self.qpi.set_alpha(np.append(np.maximum(0.001, km.fuzzy_labels_)*100, \
                        0.001 * np.ones(model.data.N)[:,np.newaxis], axis=1))


            ## factor features
            # global factors
            min_value = np.min(self.model.data.obs)
            max_value = np.max(self.model.data.obs)
            #TODO: this can be put into a function and shared with x min/max
            if model.settings.g_link == "exp":
                min_value = np.log(min_value)
                max_value = np.log(max_value)
            elif model.settings.g_link == "softplus":
                min_value = softplus_inverse(min_value)
                max_value = softplus_inverse(max_value)
            elif model.settings.g_link == "sigmoid":
                if min_value < 1e-100:
                    min_value = -np.inf
                else:
                    min_value = -np.log(1./min_value - 1)
                if max_value > 1-1e-10:
                    max_value = np.inf
                else:
                    max_value = -np.log(1./max_value - 1)
            elif model.settings.g_link == "expinverse":
                min_value = np.log(min_value ** -1)
                max_value = np.log(max_value ** -1)
            self.qmu = NormalParameter(min_value, max_value)
            self.qsigma = iWishartParameter()
            if merge is not None:
                beta = model.submodel.qbeta.mean()
                init_mu = model.submodel.qmu.loc
                init_sigma_df = model.submodel.qsigma.df
                init_sigma_scale = model.submodel.qsigma.scale

                # normalize beta
                beta = beta / np.sum(beta)

                init_mu[merge[0]] = (init_mu[merge[0]] * beta[merge[0]] + \
                        init_mu[merge[1]] * beta[merge[1]]) / \
                        (beta[merge[0]] + beta[merge[1]])
                init_sigma_df[merge[0]] = (init_sigma_df[merge[0]] * beta[merge[0]] + \
                        init_sigma_df[merge[1]] * beta[merge[1]]) / \
                        (beta[merge[0]] + beta[merge[1]])
                init_sigma_scale[merge[0]] = (init_sigma_scale[merge[0]] * beta[merge[0]] + \
                        init_sigma_scale[merge[1]] * beta[merge[1]]) / \
                        (beta[merge[0]] + beta[merge[1]])
                self.qmu.set_loc(np.delete(init_mu, merge[1], axis=0))
                self.qsigma.set_df(np.delete(init_sigma_df, merge[1], axis=0))
                self.qsigma.set_scale(np.delete(init_sigma_scale, merge[1], axis=0))
            elif split is not None:
                init_mu = model.submodel.qmu.loc
                init_sigma_scale = model.submodel.qsigma.scale
                init_sigma_df = model.submodel.qsigma.df

                #self.qmu.set_loc(np.insert(init_mu, -1, \
                #        init_mu[split,np.newaxis] + \
                #        np.random.normal(0, np.sqrt(np.var(init_mu,0))/self.K, init_mu[split].shape), axis=0))

                init_mu = np.insert(init_mu, -1, km.cluster_centers_[1], axis=0)
                init_mu[split] = km.cluster_centers_[0]
                self.qmu.set_loc(init_mu)

                self.qsigma.set_df(np.append(init_sigma_df, \
                        init_sigma_df[split,np.newaxis], axis=0))
                self.qsigma.set_scale(np.append(init_sigma_scale, \
                        init_sigma_scale[split,np.newaxis], axis=0))
            else:
                cluster_centers = km.cluster_centers_
                # transform
                if self.model.settings.g_link == "exp":
                    cluster_centers = np.log(cluster_centers)
                elif self.model.settings.g_link == "softplus":
                    cluster_centers = softplus_inverse(cluster_centers)
                elif self.model.settings.g_link == "sigmoid":
                    cluster_centers[cluster_centers < 1e-100] = 1e-100
                    cluster_centers[cluster_centers > 1-1e-10] = 1 - 1e-10
                    cluster_centers = -np.log(1./cluster_centers - 1)
                elif self.model.settings.g_link == "expinverse":
                    cluster_centers = np.log(cluster_centers ** -1)

                if model.settings.fix_K:
                    self.qmu.set_loc(cluster_centers)
                    self.qsigma.set_scale(np.array(np.zeros([K, model.data.M, \
                        model.data.M])))
                    data_var = np.var(model.data.transformed_obs, 0)
                    for k in range(K):
                        self.qsigma.scale[k] = np.identity(model.data.M) * data_var
                    self.qsigma.scale *= model.data.M
                    self.qsigma.set_df(np.array([model.data.M + 1 + 1e6] * K))
                else:
                    self.qmu.set_loc(np.append(cluster_centers, \
                            np.mean(model.data.transformed_obs, 0)[np.newaxis], \
                            axis=0))
                    self.qsigma.set_scale(np.array(np.zeros([K+1, model.data.M, \
                        model.data.M])))
                    data_var = np.var(model.data.transformed_obs, 0)
                    for k in range(K+1):
                        self.qsigma.scale[k] = np.identity(model.data.M) * data_var
                    self.qsigma.scale *= model.data.M
                    self.qsigma.set_df(np.array([model.data.M + 1 + 1e6] * (K+1)))
            #TODO: set var on mu?
            self.qmu.set_var(1e-5)


            # local features
            min_value = min(np.min(self.model.data.transformed_obs) * 2, \
                    np.min(self.model.data.transformed_obs) / 2)
            max_value = max(np.max(self.model.data.transformed_obs) * 2, \
                    np.max(self.model.data.transformed_obs) / 2)
            self.qx = NormalParameter(min_value, max_value)
            if merge is not None:
                pi = model.submodel.qpi.alpha
                init_loc = model.submodel.qx.loc
                init_var = model.submodel.qx.var
                init_loc[:,merge[0]] = \
                        (init_loc[:,merge[0]] * pi[:,merge[0],np.newaxis] + \
                        init_loc[:,merge[1]] * pi[:,merge[1],np.newaxis]) / \
                        (pi[:,merge[0],np.newaxis] + pi[:,merge[1],np.newaxis])
                init_var[:,merge[0]] = \
                        (init_var[:,merge[0]] * pi[:,merge[0],np.newaxis] + \
                        init_var[:,merge[1]] * pi[:,merge[1],np.newaxis]) / \
                        (pi[:,merge[0],np.newaxis] + pi[:,merge[1],np.newaxis])
                self.qx.set_loc(np.delete(init_loc, merge[1], axis=1))
                self.qx.set_var(np.delete(init_var, merge[1], axis=1))
            elif split is not None:
                rho = (model.iteration + 4) ** -0.5
                init_loc = model.submodel.qx.loc
                init_var = model.submodel.qx.var

                #self.qx.set_loc(np.append(init_loc, \
                #        np.ones([model.data.N, 1, model.data.M]) * self.qmu.loc[-2] + \
                #        (init_loc[:,np.newaxis,split] - self.qmu.loc[np.newaxis,split]), axis=1))
                init_loc = np.append(init_loc, np.ones([model.data.N, 1, model.data.M]) * self.qmu.loc[-2], axis=1)
                abs_diff = np.sqrt((init_loc[:,np.newaxis,split] - self.qmu.loc[(split, -2),])**2).sum(2)

                init_loc[abs_diff[:,0] > abs_diff[:,1], -2] = init_loc[abs_diff[:,0] > abs_diff[:,1], split]
                init_loc[abs_diff[:,0] > abs_diff[:,1], split] = self.qmu.loc[split]

                #
                pi = self.qpi.mean()
                residuals = (model.data.transformed_obs - \
                        ((init_loc * pi[:,:,np.newaxis]).sum(1) - \
                        pi[:,-1,np.newaxis] * self.qmu.loc[-1]))# / \
                #        pi[:,-1,np.newaxis]
                init_loc[:, K] = residuals
                #

                self.qx.set_loc(init_loc)

                self.qx.set_var(np.append(init_var, \
                        init_var[:,split,np.newaxis], axis=1))
            else:
                if model.settings.fix_K:
                    # init to mu
                    self.qx.set_loc(np.ones([model.data.N, K, model.data.M]) * self.qmu.loc)
                    self.qx.set_var(np.var(model.data.transformed_obs, 0) * 1e-5 * \
                            np.ones([model.data.N, K, model.data.M]))
                else:
                    init_loc = np.ones([model.data.N, K+1, model.data.M]) * self.qmu.loc
                    pi = self.qpi.mean()
                    #residuals = (model.data.transformed_obs - \
                    #        ((init_loc * pi[:,:,np.newaxis]).sum(1) - \
                    #        pi[:,-1,np.newaxis] * self.qmu.loc[-1])) / \
                    #        pi[:,-1,np.newaxis]
                    residuals = (model.data.transformed_obs - \
                            ((init_loc * pi[:,:,np.newaxis]).sum(1) - \
                            pi[:,-1,np.newaxis] * self.qmu.loc[-1]))
                    init_loc[:, K] = residuals
                    self.qx.set_loc(init_loc)
                    self.qx.set_var(np.var(model.data.transformed_obs, 0) * 1e-5 * \
                            np.ones([model.data.N, K+1, model.data.M]))

                    # update mu for k>K
                    loc = self.qmu.loc
                    loc[K] = np.mean(residuals)
                    self.qmu.set_loc(loc)

            # local counts
            self.qP = PoissonParameter()
            if model.data.known_density:
                self.qP.set_rate(model.data.density)
            else:
                if merge is not None or split is not None:
                    self.qP.set_rate(model.submodel.qP.rate)
                else:
                    #PTO (TODO?)
                    if self.model.settings.f_dist == "poisson":
                        model.settings.rho = np.max(model.data.obs, 1)

                    #TODO: this needs to be set with care for each kind of application, possibly fixed to data when we know P
                    self.qP.set_rate(np.ones(model.data.N) * model.settings.rho)

            self.reset_MS = True


        def loglikelihood(self):
            mu = (self.qx.mean() * np.repeat(self.qpi.mean()[:,:,np.newaxis],self.model.data.M,axis=2)).sum(1)
            #mu = self.qx.mean() * np.repeat(self.qpi.mean()[:,:,np.newaxis],self.model.data.M,axis=2)
            #if self.model.settings.fix_K:
            #    mu = mu.sum(1)
            #else:
            #    mu = mu[:,:-1].sum(1)
            return self.model.loglikelihood(self.model.data.obs, mu).sum()

        def ELBO(self):
            beta = self.qbeta.mean()
            mu = self.qmu.mean()
            Sigma, iSigma, detSigma = self.qsigma.vals()
            pi = self.qpi.mean()
            x = self.qx.mean()
            P = self.qP.mean()

            # log p
            logL = self.loglikelihood()
            rv = logL + \
                    logNormal(mu, self.model.mu_prior, self.model.sigma_prior.diagonal()**2).sum()
            #print "mu     ", (rv - logL)
            if self.model.settings.fix_K:
                rv += logDirichlet(beta, self.model.settings.gbl_alpha * np.ones(self.K))
                rv += logDirichlet(pi, beta * self.model.settings.lcl_alpha).sum()
            else:
                rv += logDPBeta(beta, self.model.settings.gbl_alpha)
                #print "beta   ", logDPBeta(beta, self.model.settings.gbl_alpha)
                #rv += logDPGamma(pi, beta * self.model.settings.lcl_alpha).sum()
                #rv += logDirichlet(np.append(pi, np.ones(pi.shape[0])[:,np.newaxis] * 1e-6, axis=1), \
                #        beta * self.model.settings.lcl_alpha).sum()
                rv += logDirichlet(pi, beta * self.model.settings.lcl_alpha).sum()
                #print "pi     ", logDirichlet(pi, beta * self.model.settings.lcl_alpha).sum()

            tmp = rv
            for k in range(self.K if self.model.settings.fix_K else self.K + 1):
                rv += logiWishart(Sigma[k], self.model.data.M, self.model.sigma_prior).sum()
            #print "sigma  ", (rv-tmp)

            tmp_x = 0
            for n in range(self.model.data.N):
                for k in range(self.K if self.model.settings.fix_K else self.K + 1):
                    rv += logMVNormal(x[n,k], mu[k], \
                            Sigma[k], iSigma[k], detSigma[k], (P[n] * pi[n,k]))
                    tmp_x += logMVNormal(x[n,k], mu[k], \
                            Sigma[k], iSigma[k], detSigma[k], (P[n] * pi[n,k]))

                if not self.model.data.known_density:
                    if self.model.settings.f_dist == "poisson":
                        rv += logPoisson(P[n], self.model.settings.rho[n]).sum()
                    else:
                        rv += logPoisson(P[n], self.model.settings.rho).sum()
            #print "x      ", tmp_x

            # log q
            tmp = rv
            rv -= self.qbeta.logq() + self.qmu.logq() + self.qsigma.logq() + \
                    self.qpi.logq() + self.qx.logq() + self.qP.logq()
            #print "-q     ", (rv-tmp)
            #print "\t", (-self.qbeta.logq(), -self.qmu.logq(), -self.qsigma.logq(), \
            #        -self.qpi.logq(), -self.qx.logq(), -self.qP.logq())

            return rv, logL

        def update_local_counts(self, iter):
            Sigma, iSigma, detSigma = self.qsigma.vals()
            pi = self.qpi.mean()
            mu = self.qmu.mean()
            x = self.qx.mean()
            counts = self.qP.mean()
            orig_counts = counts + 0.0

            # number of samples
            S = 2**4

            # Robbins-Monro sequence for step size
            rho = (iter + 2**5) ** -0.7

            # initialize MS for RMSProp
            if iter == 0 or self.reset_MS:
                self.MS_P = np.zeros(self.model.data.N)


            if self.model.settings.cores > 1:
                bb_results = []
                for n in range(self.model.data.N):
                    bb_results.append(self.model.pool.apply_async(get_local_count_pqg, \
                            (S, counts[n], \
                            self.model.settings.rho[n] if self.model.settings.f_dist == "poisson" else self.model.settings.rho, \
                            self.K, x[n], mu, Sigma, iSigma, detSigma, pi[n])))

            for n in range(self.model.data.N):
                if self.model.settings.cores > 1:
                    p, g, q = bb_results[n].get()
                else:
                    p, g, q = get_local_count_pqg(S, counts[n], \
                            self.model.settings.rho[n] if self.model.settings.f_dist == "poisson" else self.model.settings.rho, \
                            self.K, x[n], mu, Sigma, iSigma, detSigma, pi[n])

                # RMSprop: keep running average of gradient magnitudes
                # (the gradient will be divided by sqrt of this later)
                if self.MS_P[n] == 0:
                    self.MS_P[n] = (g**2).sum()
                else:
                    self.MS_P[n] = 0.9 * self.MS_P[n] + \
                            0.1 * (g**2).sum()

                cv = covar(g * (p - q), g) / var(g)
                if g.sum() != 0:
                    counts[n] = softplus(softplus_inverse(counts[n]) \
                            + rho * (1. / S) * ((g/np.sqrt(self.MS_P[n])) * (p - q - cv)).sum(0))

            self.qP.set_rate(counts)

        def update_local_factors(self, iter):
            Sigma, iSigma, detSigma = self.qsigma.vals()
            pi = self.qpi.mean()
            mu = self.qmu.mean()
            x_loc = self.qx.mean()
            x_scl = np.sqrt(self.qx.var)
            counts = self.qP.mean()

            # number of samples
            S = 2**4

            # Robbins-Monro sequence for step size
            if self.model.settings.f_dist == "beta":
                rho_loc = (iter + 2**10) ** -0.8
            else:
                rho_loc = (iter + 2**5) ** -0.8
            rho_loc = (iter + 2**20) ** -0.8
            rho_scl = (iter + 2**20) ** -0.8

            # initialize MS for RMSProp
            if iter == 0 or self.reset_MS: #TODO: split/merge MS too?
                self.MS_x_loc = np.zeros((self.model.data.N, self.K, self.model.data.M))
                self.MS_x_scl = np.zeros((self.model.data.N, self.K, self.model.data.M))

            if self.model.settings.cores > 1:
                bb_results = []
                for n in range(self.model.data.N):
                    bb_results.append(self.model.pool.apply_async(get_local_factors_bb, \
                            (self.K, S, self.model.data.M, pi[n], x_loc[n], x_scl[n], mu, Sigma, iSigma, detSigma, counts[n], \
                            self.model.data.obs[n], self.MS_x_loc[n], self.MS_x_scl[n], rho_loc, rho_scl, \
                            self.model.settings.g_link, self.model.settings.f_dist, self.model.eta)))

            for n in range(self.model.data.N):
                if self.model.settings.cores > 1:
                    loc, scl, MS_x_loc, MS_x_scl = bb_results[n].get()
                else:
                    loc, scl, MS_x_loc, MS_x_scl = get_local_factors_bb(self.K, S, \
                            self.model.data.M, pi[n], x_loc[n], x_scl[n], mu, Sigma, iSigma, detSigma, counts[n], \
                            self.model.data.obs[n], self.MS_x_loc[n], self.MS_x_scl[n], rho_loc, rho_scl, \
                            self.model.settings.g_link, self.model.settings.f_dist, self.model.eta)
                self.MS_x_loc[n] = MS_x_loc
                self.MS_x_scl[n] = MS_x_scl
                x_loc[n] = loc
                x_scl[n] = scl

            if not self.model.settings.fix_K:
                #residuals = (self.model.data.transformed_obs - \
                #        ((x_loc * pi[:,:,np.newaxis]).sum(1) - (x_loc[:,-1] * pi[:,-1,np.newaxis]))) / \
                #        pi[:,-1,np.newaxis]
                residuals = (self.model.data.transformed_obs - \
                        ((x_loc * pi[:,:,np.newaxis]).sum(1) - (x_loc[:,-1] * pi[:,-1,np.newaxis])))
                x_loc[:, -1] = residuals

            self.qx.set_loc(x_loc)
            self.qx.set_var(x_scl ** 2)

        def update_local_proportions(self, iter):
            beta = self.qbeta.mean()
            alpha = self.qpi.alpha
            pi = self.qpi.mean()
            x = self.qx.mean()
            mu = self.qmu.mean()
            Sigma, iSigma, detSigma = self.qsigma.vals()
            counts = self.qP.mean()

            ### BBVI
            # number of samples
            S = 2**4

            # Robbins-Monro sequence for step size
            rho = (iter + 2**10) ** -0.8

            # initialize MS for RMSProp
            if iter == 0 or self.reset_MS:
                self.MS_pi = np.zeros((self.model.data.N, self.K if self.model.settings.fix_K else self.K + 1))

            for n in range(self.model.data.N):
                p = np.zeros((S, self.K if self.model.settings.fix_K else self.K + 1))
                q = np.zeros((S, self.K if self.model.settings.fix_K else self.K + 1))
                g = np.zeros((S, self.K if self.model.settings.fix_K else self.K + 1))

                for i in range(S):
                    # sample pi
                    s = sampleDirichlet(alpha[n])

                    p[i] += logDirichlet(s, self.model.settings.lcl_alpha * beta)
                    p[i] += self.model.loglikelihood(self.model.data.obs[n], np.matmul(s, x[n])).sum()

                    for k in range(self.K if self.model.settings.fix_K else self.K + 1):
                        try:
                            p[i] += logMVNormal(x[n,k], mu[k], \
                                    Sigma[k], iSigma[k], detSigma[k], (counts[n] * s[k]))
                        except:
                            print(Sigma[k])
                            print(counts[n])
                            print(s[k])
                            raise

                    q[i] = logDirichlet(s, alpha[n])
                    g[i] = loggDirichlet(s, alpha[n]) * softplus_derivative(softplus_inverse(alpha[n]))

                # RMSprop: keep running average of gradient magnitudes
                # (the gradient will be divided by sqrt of this later)
                if self.MS_pi[n].all() == 0:
                    self.MS_pi[n] = (g**2).sum(0)
                else:
                    self.MS_pi[n] = 0.9 * self.MS_pi[n] + \
                            0.1 * (g**2).sum(0)

                cv = covar(g * (p - q), g) / var(g)
                alpha[n] = softplus(softplus_inverse(alpha[n]) + \
                        rho * (1. / S) * ((g/np.sqrt(self.MS_pi[n])) * (p - q - cv)).sum(0))

            self.qpi.set_alpha(alpha)


        def update_global_factors(self):
            inv_sigma = np.linalg.inv(self.qsigma.mean())
            x_scales = self.qP.mean()[:,np.newaxis] * self.qpi.mean()
            sum_x = np.sum(self.qx.mean() * x_scales[:,:,np.newaxis], 0)
            new_mu = np.zeros((self.K if self.model.settings.fix_K else self.K + 1, self.model.data.M))
            for k in range(self.K if self.model.settings.fix_K else self.K + 1):
                new_mu[k] = (self.model.mu_sigma_prior**-1 * self.model.mu_prior + \
                        inv_sigma[k].diagonal() * sum_x[k]) / \
                        (self.model.mu_sigma_prior**-1 + inv_sigma[k].diagonal() * x_scales.sum(0)[k])

            new_sigma_df = self.model.data.M + x_scales.sum(0)
            new_sigma_scale = np.repeat(self.model.sigma_prior[np.newaxis], \
                    self.K if self.model.settings.fix_K else self.K + 1, axis=0)
            diff = (self.qx.mean() - new_mu)
            for k in range(self.K if self.model.settings.fix_K else self.K + 1):
                new_sigma_scale[k] = new_sigma_scale[k] + \
                        np.matmul(diff[:,k].T, diff[:,k] * x_scales[:,k,np.newaxis])

            self.qmu.set_loc(new_mu)
            self.qsigma.set_df(new_sigma_df)
            self.qsigma.set_scale(new_sigma_scale)

        def update_global_proportions(self, iter):
            alpha = self.qbeta.alpha
            pi = self.qpi.mean()

            ### BBVI
            # number of samples
            S = 2**6

            # Robbins-Monro sequence for step size
            rho = (iter + 2**4) ** -0.5
            if self.model.settings.f_dist == "normal":
                rho = (iter + 2**6) ** -0.5
            print(rho)
            #print(rho)


            # initialize MS for RMSProp
            if iter == 0 or self.reset_MS:
                self.MS_beta = np.zeros(self.K)
                self.reset_MS = False

            p = np.zeros((S, self.K if self.model.settings.fix_K else self.K+1))
            q = np.zeros((S, self.K if self.model.settings.fix_K else self.K+1))
            g = np.zeros((S, self.K if self.model.settings.fix_K else self.K+1))

            for i in range(S):
                # sample beta
                s = sampleDirichlet(alpha)

                p[i] += logDirichlet(s, self.model.settings.gbl_alpha * \
                        np.ones(self.K if self.model.settings.fix_K else self.K+1))
                for n in range(self.model.data.N):
                    p[i] += logDirichlet(pi[n], self.model.settings.lcl_alpha * s)

                q[i] = logDirichlet(s, alpha)
                g[i] = loggDirichlet(s, alpha) * softplus_derivative(softplus_inverse(alpha))

            # RMSprop: keep running average of gradient magnitudes
            # (the gradient will be divided by sqrt of this later)
            if self.MS_beta.all() == 0:
                self.MS_beta = (g**2).sum(0)
            else:
                self.MS_beta = 0.9 * self.MS_beta + \
                        0.1 * (g**2).sum(0)

            cv = covar(g * (p - q), g) / var(g)
            alpha = softplus(softplus_inverse(alpha) + rho * (1. / S) * ((g/np.sqrt(self.MS_beta)) * (p - q - cv)).sum(0))

            self.qbeta.set_alpha(alpha)


        def update(self):
            self.update_local_factors(self.model.iteration)
            self.update_local_proportions(self.model.iteration)
            if not self.model.data.known_density:
                self.update_local_counts(self.model.iteration)
            self.update_global_factors()
            self.update_global_proportions(self.model.iteration)

    def adopt_submodel(self, submodel):
        self.submodel = submodel

    def fit(self):
        converged = False
        self.iteration = 0
        fout = open(os.path.join(self.settings.outdir, 'logfile.csv'), 'w+')
        fout.write("iteration,ELBO,loglikelihood\n")
        old_ELBO, old_logL = self.submodel.ELBO()
        max_ELBO = old_ELBO
        print("initialization\tELBO: %f\tlogL: %f" % (old_ELBO, old_logL))
        fout.write("0,%f,%f\n" % (old_ELBO, old_logL))
        convergence_counter = 0
        batch_iteration_counter = 0
        self.movie_counter = 0
        self.movie_out = open(os.path.join(self.settings.outdir, 'movie_centers.csv'), 'w+')
        self.movie_out.write("iter,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10\n")
        self.movie_x = open(os.path.join(self.settings.outdir, 'movie_x.csv'), 'w+')
        self.movie_x.write("iter,K,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10\n")
        mu = self.submodel.qmu.mean()
        x = self.submodel.qx.mean()
        for k in range(self.submodel.K if self.settings.fix_K else self.submodel.K+1):
            self.movie_out.write("%d,%s\n" % \
                    (self.movie_counter, \
                    ','.join(str(i) for i in mu[k])))
            for n in range(self.data.N):
                self.movie_x.write("%d,%d,%s\n" % \
                        (self.movie_counter, k, \
                        ','.join(str(i) for i in x[n,k])))
        self.movie_counter += 1

        #self.save("init")
        while not converged:
            ## run mini batch of inference
            batch_converged = False
            if not self.settings.fix_K:
                print("local updates")
            while not batch_converged:
                self.submodel.update()
                #logL = self.submodel.loglikelihood()
                ELBO, logL = self.submodel.ELBO()
                self.iteration += 1
                batch_iteration_counter += 1
                fout.write("%d,%f,%f\n" % (self.iteration, ELBO, logL))

                # check for batch convergence
                #if self.settings.f_dist != "poisson" and ((old_logL - logL)/old_logL) < 1e-3:
                if ELBO < max_ELBO:
                    batch_converged = True
                    convergence_counter += 1
                elif self.settings.f_dist != "poisson" and ((old_ELBO - ELBO)/old_ELBO) < 1e-3:
                    batch_converged = True
                    convergence_counter += 1
                #elif ((old_logL - logL)/old_logL) < 1e-4:
                elif ((old_ELBO - ELBO)/old_ELBO) < 1e-4:
                    batch_converged = True
                    convergence_counter += 1
                elif np.isnan(logL) or self.iteration == self.settings.max_iter or \
                        (not self.settings.fix_K and batch_iteration_counter == self.settings.batch_max_iter):
                    batch_converged = True
                else:
                    convergence_counter = 0
                # force split/merge
                #batch_converged = True

                if self.iteration % self.settings.save_freq == 0:
                    self.save("%04d" % self.iteration)
                    mu = self.submodel.qmu.mean()
                    x = self.submodel.qx.mean()
                    for k in range(self.submodel.K if self.settings.fix_K else self.submodel.K+1):
                        self.movie_out.write("%d,%s\n" % \
                                (self.movie_counter, \
                                ','.join(str(i) for i in mu[k])))
                        for n in range(self.data.N):
                            self.movie_x.write("%d,%d,%s\n" % \
                                    (self.movie_counter, k, \
                                    ','.join(str(i) for i in x[n,k])))
                    self.movie_counter += 1

                direction = "0"
                if old_ELBO > ELBO:
                    direction = "-"
                elif old_ELBO < ELBO:
                    direction = "+"

                if old_logL > logL:
                    direction += " -"
                elif old_logL < logL:
                    direction += " +"
                else:
                    direction += ' 0'
                print("iteration %d\tELBO: %f\tlogL: %f\t%s  %d" % \
                        (self.iteration, ELBO, logL, direction, convergence_counter))
                old_logL = logL
                old_ELBO = ELBO
                if ELBO > max_ELBO:
                    max_ELBO = ELBO

            # only split/merge for free K, and don't do it if the max iteration was hit
            # (lest a split or merge be introduced but don't get more iterations to converge)
            if not self.settings.fix_K and self.iteration != self.settings.max_iter:
                print("split/merge")

                #ELBO, logL = self.submodel.ELBO()
                #print "current ELBO:\t%f" % ELBO

                ## merge
                # generate candidate merge pairs
                pairs = {}
                pi = self.submodel.qpi.mean()
                for k1 in range(self.submodel.K):
                    for k2 in range(k1+1, self.submodel.K):
                        cv = np.cov(pi[:,k1], pi[:,k2])
                        if cv[0,1] > 0: # don't bother trying to merge factors with no covariance
                            pairs[(k1, k2)] = cv[0,1] / np.sqrt(cv[0,0] * cv[1,1])

                # check if merge improves ELBO; if so, accept
                merged = set()
                merge_count = 0
                for pair in sorted(pairs, key=lambda x: -pairs[x]):
                    # can't merge factors that no longer exist due to past merges
                    if pair[0] in merged or pair[1] in merged:
                        continue

                    # don't merge too much at once
                    if merge_count >= self.submodel.K / 2:
                        print("merge cap hit (no more than %d merges allowed with %d starting factors)" % \
                                (self.submodel.K / 2, self.submodel.K))
                        continue

                    # must adjust indexes based on past merges
                    pair_orig = pair
                    pair = (pair[0] - sum(i < pair[0] for i in merged), \
                            pair[1] - sum(i < pair[1] for i in merged))

                    submodel = self.Submodel(self, K=self.submodel.K-1, \
                        merge=pair)
                    submodel.update()

                    merge_ELBO, merge_logL = submodel.ELBO()
                    print(merge_ELBO, merge_logL)

                    if merge_ELBO > ELBO:
                        ELBO = merge_ELBO
                        logL = merge_logL
                        old_logL = logL
                        old_ELBO = ELBO
                        if ELBO > max_ELBO:
                            max_ELBO = ELBO
                        self.adopt_submodel(submodel)
                        merge_count += 1
                        print("MERGE: ADOPTING SUBMODEL WITH K=", \
                            self.submodel.K, ELBO, logL)
                        fout.write("%dm,%f,%f\n" % (self.iteration, ELBO, logL))
                        convergence_counter = 0
                        merged.add(pair_orig[1])
                        mu = self.submodel.qmu.mean()
                        x = self.submodel.qx.mean()
                        for k in range(self.submodel.K if self.settings.fix_K else self.submodel.K+1):
                            self.movie_out.write("%d,%s\n" % \
                                    (self.movie_counter, \
                                    ','.join(str(i) for i in mu[k])))
                            for n in range(self.data.N):
                                self.movie_x.write("%d,%d,%s\n" % \
                                        (self.movie_counter, k, \
                                        ','.join(str(i) for i in x[n,k])))
                        self.movie_counter += 1
                print("%d out of %d candidate merges completed" % (merge_count, len(pairs)))

                ## split
                for k in sorted(range(self.submodel.K), key=lambda x: np.random.random()):
                    submodel = self.Submodel(self, K=self.submodel.K+1, split=k)
                    submodel.update()
                    #print "after update"
                    #print submodel.qmu.loc

                    split_ELBO, split_logL = submodel.ELBO()
                    print(split_ELBO, split_logL)

                    if split_ELBO > ELBO:
                        ELBO = split_ELBO
                        logL = split_logL
                        old_logL = logL
                        old_ELBO = ELBO
                        if ELBO > max_ELBO:
                            max_ELBO = ELBO
                        self.adopt_submodel(submodel)
                        print("SPLIT: adopting submodel with K=", self.submodel.K, \
                            ELBO, logL)
                        fout.write("%ds,%f,%f\n" % (self.iteration, ELBO, logL))
                        convergence_counter = 0

                        mu = self.submodel.qmu.mean()
                        x = self.submodel.qx.mean()
                        for k in range(self.submodel.K if self.settings.fix_K else self.submodel.K+1):
                            self.movie_out.write("%d,%s\n" % \
                                    (self.movie_counter, \
                                    ','.join(str(i) for i in mu[k])))
                            for n in range(self.data.N):
                                self.movie_x.write("%d,%d,%s\n" % \
                                        (self.movie_counter, k, \
                                        ','.join(str(i) for i in x[n,k])))
                        self.movie_counter += 1


            # check for convergence
            batch_iteration_counter = 0
            if self.iteration == self.settings.max_iter:
                converged = True
                print("* Maximum number of iterations reached")
            elif np.isnan(logL):
                converged = True
                print("* Encountered NaNs")
            elif self.iteration >= self.settings.min_iter:
                if convergence_counter > 3:
                    converged = True
                    print("* Data likelihood converged")

        self.save("final")
        mu = self.submodel.qmu.mean()
        x = self.submodel.qx.mean()
        for k in range(self.submodel.K if self.settings.fix_K else self.submodel.K+1):
            self.movie_out.write("%d,%s\n" % \
                    (self.movie_counter, \
                    ','.join(str(i) for i in mu[k])))
            for n in range(self.data.N):
                self.movie_x.write("%d,%d,%s\n" % \
                        (self.movie_counter, k, \
                        ','.join(str(i) for i in x[n,k])))
        self.movie_counter += 1
        self.movie_out.close()
        self.movie_x.close()

    def save(self, label):
        fname = os.path.join(self.settings.outdir, 'model-%s.hdf5' % label)
        f = open(fname, 'w+')
        f.close()

        f = h5py.File(fname, 'w')

        f.create_dataset("global_factor_concentration", \
                data=self.submodel.qbeta.mean())

        f.create_dataset("local_factor_concentration", \
                data=self.submodel.qpi.mean())

        f.create_dataset("local_factor_counts", \
                data=self.submodel.qP.mean())

        f.create_dataset("global_factor_features_mean", \
                data=self.submodel.qmu.mean())
        f.create_dataset("global_factor_features_cov", \
                data=self.submodel.qsigma.mean())
        f.create_dataset("local_factor_features", \
                data=self.submodel.qx.mean())

        f.close()

        if self.settings.overwrite:
            if self.last_label != '':
                os.remove(os.path.join(self.settings.outdir, \
                    'model-%s.hdf5' % self.last_label))
            self.last_label = label
