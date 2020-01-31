import os, subprocess
from datetime import datetime as dt


class Settings:
    def __init__(self, seed, outdir, save_freq, save_all, conv_thresh, \
        min_iter, max_iter, batch_max_iter, tau, kappa, sample_size, K, fix_K, \
        gbl_alpha, lcl_alpha, rho, f_dist, g_link, cores):
        self.seed = seed
        self.outdir = outdir
        self.save_freq = save_freq
        self.overwrite = not save_all
        self.convergence_thresh = conv_thresh
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.batch_max_iter = batch_max_iter
        self.tau = tau # delay
        self.kappa = kappa # forgetting rate
        self.sample_size = sample_size # BBVI samples S
        self.cores = cores

        self.K = K
        self.fix_K = fix_K

        self.gbl_alpha = gbl_alpha
        self.lcl_alpha = lcl_alpha
        self.rho = rho

        if f_dist not in ['normal', 'log_normal', 'gamma', \
                'exponential', 'poisson', 'beta']:
            raise ValueError('f distribution must be one of: normal, '
                    'log_normal, gamma, exponential, poisson, beta')
        self.f_dist = f_dist

        if g_link == 'none':
            if f_dist in ['normal', 'log_normal']:
                g_link = 'identity'
            elif f_dist in ['gamma', 'poisson']:
                g_link = 'softplus'
            elif f_dist in ['beta']:
                g_link = 'sigmoid'
            elif f_dist in ['exponential']:
                g_link = 'expinverse'
        if g_link not in ['identity', 'exp', 'softplus', 'sigmoid', \
                'expinverse']:
            raise ValueError('g link function must be one of: identity, '
                    'exp, softplus, sigmoid, expinverse')
        self.g_link = g_link

    def save(self, datadir, message):
        f = open(os.path.join(self.outdir, 'settings.dat'), 'w+')

        f.write("%s\n" % dt.now())
        f.write("%s\n\n" % message)

        f.write("data dir:\t%s\n\n" % datadir)

        f.write("random seed:\t%d\n" % self.seed)
        f.write("save frequency:\t%d\n" % self.save_freq)
        f.write("overwrite:\t%s\n" % ('True' if self.overwrite else 'False'))
        f.write("convergence threshold:\t%f\n" % self.convergence_thresh)
        f.write("min # of iterations:\t%d\n" % self.min_iter)
        f.write("max # of iterations:\t%d\n" % self.max_iter)
        f.write("max # of batch iterations:\t%d\n" % self.batch_max_iter)
        f.write("delay tau:\t%d\n" % self.tau)
        f.write("forgetting rate kappa:\t%f\n" % self.kappa)
        f.write("BBVI sample size:\t%d\n\n" % self.sample_size)

        f.write("initial K:\t%d\n" % self.K)
        f.write("%s\n\n" % ("Fixed K (parametric)" if self.fix_K else \
            "Learned K (nonparametric)"))

        f.write("global concentration:\t%f\n" % self.gbl_alpha)
        f.write("local concentration:\t%f\n" % self.lcl_alpha)
        f.write("local counts prior:\t%f\n" % self.rho)
        f.write("distribution f for observations y:\t%s\n\n" % self.f_dist)
        f.write("link function g for observations y:\t%s\n\n" % self.g_link)

        p = subprocess.Popen(['git','rev-parse', 'HEAD'], \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        f.write("\ncommit #%s" % out)

        p = subprocess.Popen(['git','diff'], stdout=subprocess.PIPE, \
            stderr=subprocess.PIPE)
        out, err = p.communicate()
        f.write("%s" % out)

        f.close()
