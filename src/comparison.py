import h5py
import numpy as np
import argparse, os, shutil

# set up argument parser and parse args
parser = argparse.ArgumentParser(description = \
    'This script converts from data from the NDM-required HDF5 format to '
    'the various format required by the code for the comparison methods.')
parser.add_argument('--in', dest='infile', type=str, \
    help='the path and filename of the HDF5 input file')
parser.add_argument('--out', dest='outdir', type=str, \
    help='the output idirectory in which to write the various output files')
parser.add_argument('--K', dest='K', type=int, default = 10, help = '# of latent factors')

parser.add_argument('--nmf', dest='NMF_bool', action='store_true', help='Toggle Run NFM')
parser.add_argument('--lda', dest='LDA_bool', action='store_true', help='Toggle Run LDA')
parser.add_argument('--pf', dest='PF_bool', action='store_true', help='Toggle Run Poisson Factorization')
parser.add_argument('--pmf', dest='PMF_bool', action='store_true', help='Toggle Run PMF')

args = parser.parse_args()

# strip filename for output prefix
out_prefix = args.infile.split('.')[0]

# create output dir (check if exists)
if os.path.exists(args.outdir):
    print "Output directory %s already exists.  Removing it to have a clean output directory!" % args.outdir
    shutil.rmtree(args.outdir)
os.makedirs(args.outdir)

# create output file
fout = h5py.File(os.path.join(args.outdir, 'results.hdf5'), 'w')

# read in the observed data
f = h5py.File(args.infile, 'r')
obs = np.array(f['observations'])
N = obs.shape[0] # num observtions
M = obs.shape[1] # num features



# PCA
from sklearn.decomposition import PCA

pca = PCA(n_components = args.K)
pca_z = pca.fit_transform(obs)
fout.create_dataset("pca/z", data = pca_z)
fout.create_dataset("pca/components", data = pca.components_)
fout.create_dataset("pca/explained_variance", data = pca.explained_variance_)
fout.create_dataset("pca/noise_variance", data = pca.noise_variance_)

# Factor Analysis
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components = args.K)
fa_z = fa.fit_transform(obs)
fout.create_dataset("fa/z", data = fa_z)
fout.create_dataset("fa/components", data = fa.components_)
fout.create_dataset("fa/noise_variance", data = fa.noise_variance_)

# NMF
if args.NMF_bool:
    from sklearn.decomposition import NMF

    nmf = NMF(n_components = args.K)
    nmf_z = nmf.fit_transform(obs)
    fout.create_dataset("nmf/z", data = nmf_z)
    fout.create_dataset("nmf/components", data = nmf.components_)

# LDA
if args.LDA_bool:
    from sklearn.decomposition import LatentDirichletAllocation

    lda = LatentDirichletAllocation(n_components = args.K)
    lda_z = lda.fit_transform(obs)
    fout.create_dataset("lda/z", data = lda_z)
    fout.create_dataset("lda/components", data = lda.components_)

# Poisson Factorization
def pf_train(data, K, num_iters):
    long_data = list()
    R = data.shape[0]
    C = data.shape[1]
    for row in range(R):
        for col in range(C):
            long_data.append((row, col, data[row,col]))

    prior = 0.1
    row_rep = np.random.gamma(1, 1, (R, K))
    col_rep = np.random.gamma(1, 1, (C, K))
    for it in range(num_iters):
        new_row_rep = np.ones((R, K)) * prior
        new_col_rep = np.ones((C, K)) * prior
        for r, c, v in long_data:
            z = row_rep[r] * col_rep[c]
            z /= sum(z)
            z *= v
            new_row_rep[r] += z
            new_col_rep[c] += z

        new_row_rep /= prior + np.sum(col_rep, 0)
        col_rep = new_col_rep / (prior + np.sum(row_rep, 0))
        row_rep = new_row_rep

    return row_rep, col_rep

if args.PF_bool:
    pf_z, pf_components = pf_train(obs, args.K, 1000)
    fout.create_dataset("pf/z", data = pf_z)
    fout.create_dataset("pf/components", data = pf_components.T)

# K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = args.K)
k_z = kmeans.fit_predict(obs)
k_components = kmeans.cluster_centers_
fout.create_dataset("kmeans/z", data = k_z)
fout.create_dataset("kmeans/components", data = k_components)

# Fuzzy K-Means
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
fkmeans = FuzzyKMeans(k=args.K, m=2)
fkmeans.fit(obs)
fk_z = fkmeans.fuzzy_labels_
fk_components = fkmeans.cluster_centers_
fout.create_dataset("fuzzy_kmeans/z", data = fk_z)
fout.create_dataset("fuzzy_kmeans/components", data = fk_components)

# GMM
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = args.K)
gmm.fit(obs)
gmm_z = gmm.predict_proba(obs)
fout.create_dataset("gmm/z", data = gmm_z)
fout.create_dataset("gmm/components", data = gmm.means_)
fout.create_dataset("gmm/covariances", data = gmm.covariances_)
fout.create_dataset("gmm/weights", data = gmm.weights_)

# Probablistic Matrix Factorization
from ProbabilisticMatrixFactorization import PMF

# reshape data for input

if args.PMF_bool:
    N = obs.shape[0]*obs.shape[1]
    x = np.arange(0,obs.shape[0], 1)
    y = np.arange(0,obs.shape[1], 1)
    xv, yv = np.meshgrid(x,y)
    l3 = np.reshape(obs, (N,1))
    l2 = np.reshape(yv.T, (N,1))
    l1 = np.reshape(xv.T, (N,1))
    Y = np.concatenate((l1,l2,l3), axis = 1)

    pmf = PMF()
    pmf.set_params({"num_feat": args.K})
    pmf.fit(Y,Y)
    fout.create_dataset("pmf/z", data = pmf.w_User)
    fout.create_dataset("pmf/components", data = pmf.w_Item.T)
fout.close()
