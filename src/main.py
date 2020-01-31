from datetime import datetime as dt
import numpy as np
import shutil, os

from settings import Settings
from data import Data
from model import Model
from evaluation import Evaluation

if __name__ == '__main__':
    ## Start by parsing the arguments
    import argparse

    # general script description
    parser = argparse.ArgumentParser(description = \
        'This script identifies latent factors and uses them to deconvolve '
        'observed data into its constituent parts.')

    parser.add_argument('--data', dest='data', type=str, required=True, \
        help='a path to an HDF5 data file; see README for structure')
    parser.add_argument('--out', dest='outdir', type=str, \
        default='out', help='output directory')
    parser.add_argument('--msg', dest='message', type=str, \
        default='', help='log message')

    parser.add_argument('--save_freq', dest='save_freq', type=int, \
        default=10, help='frequency of saving current status, default 10')
    parser.add_argument('--save_all', action='store_true', \
        help='do not overwrite intermediate saved states')
    parser.add_argument('--conv_thresh', dest='conv_thresh', type=float, \
        default=1e-4, help='convergence threshold to stop fit, default 1e-4')
    parser.add_argument('--min_iter', dest='min_iter', type=int, \
        default=25, help='minimum number of iterations, default 25')
    parser.add_argument('--max_iter', dest='max_iter', type=int, \
        default=1000, help='maximum number of iterations, default 1000')
    parser.add_argument('--batch_max_iter', dest='batch_max_iter', type=int, \
        default=20, help='maximum number of iterations per batch (nonparametric only), default 20')
    parser.add_argument('--tau', dest='tau', type=int, \
        default=2**10, help='stochastic gradient delay; default 1024')
    parser.add_argument('--kappa', dest='kappa', type=float, \
        default=0.7, help='stochastic gradient forgetting rate; default 0.7')
    parser.add_argument('--sample_size', dest='sample_size', type=int, \
        default=2**6, help='stochastic gradient sample size; default 64')
    parser.add_argument('--seed', dest='seed', type=int, \
        default=(dt.fromtimestamp(0) - dt.now()).microseconds, \
        help='random seed, default from time')
    parser.add_argument('--cores', dest='cores', type=int, \
        default=1, help='number of cores to use')

    parser.add_argument('--K', dest='K', type=int, \
        default=10, help='initial (if nonparametric) or fixed K, default 10')
    parser.add_argument('--fix_K', dest='fix_K', action='store_true', \
        help='fix the number of latent components K; default off '
        '(nonparametric/learned K)')

    parser.add_argument('--gbl_con', dest='gbl_alpha', type=float, \
        default=1.0, help='global concentration parameter, default 1.0')
    parser.add_argument('--lcl_con', dest='lcl_alpha', type=float, \
        default=10.0, help='global concentration parameter, default 10.0')

    parser.add_argument('--rho', dest='rho', type=float, \
        default=1.0, help='local counts prior, default 1.0')

    parser.add_argument('--dist', dest='f_dist', type=str, \
        default='normal', help='distribution f for observtions y, '
        'default normal.  options: normal, log_normal, gamma, '
        'exponential, poisson, beta')
    parser.add_argument('--link', dest='g_link', type=str, \
        default='none', help='link function for observations y,'
        'default depends on f distribution.  options: identity '
        '(normal, log_normal), exp, softplus (gamma, poisson),'
        'sigmoid (beta), expinverse (exponential)')

    # parse the arguments
    args = parser.parse_args()

    ## Other setup: input (data), output, parameters object
    # create output dir (check if exists)
    if os.path.exists(args.outdir):
        print("Output directory %s already exists.  Removing it to have a clean output directory!" % args.outdir)
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    # create an object of model settings
    settings = Settings(args.seed, args.outdir, \
        args.save_freq, args.save_all, \
        args.conv_thresh, args.min_iter, args.max_iter, args.batch_max_iter, \
        args.tau, args.kappa, args.sample_size, \
        args.K, args.fix_K, \
        args.gbl_alpha, args.lcl_alpha, \
        args.rho, \
        args.f_dist, args.g_link, args.cores)
    settings.save(args.data, args.message)

    # read in data
    data = Data(args.data)

    ## Fit model
    model = Model(settings, data)
    model.fit()
