import os, sys, h5py
import numpy as np

if len(sys.argv) != 2:
    print "usage: python collapse_model_results.py [fit directory]"
    sys.exit(-1)

directory = sys.argv[1]
fout = open(os.path.join(directory, 'collapsed_model_results.csv'), 'w+')
fout.write("iteration,parameter,K,N,M,value\n")
for filename in os.listdir(directory):
    if not filename.endswith(".hdf5"):
        continue
    if 'final' in filename:
        continue

    print "processing", filename, "..."
    iteration = int(filename.split('.')[0].split('-')[-1])
    f = h5py.File(os.path.join(directory, filename), 'r')

    beta = np.array(f['global_concentration'])
    pi = np.array(f['local_concentration'])
    eta = np.array(f['global_features'])
    x = np.array(f['local_features'])

    K = beta.shape[0]
    N = pi.shape[0]
    M = eta.shape[1]

    for k in range(K):
        fout.write("%d,beta,%d,,,%f\n" % (iteration, k, beta[k]))
        for n in range(N):
            fout.write("%d,pi,%d,%d,,%f\n" % (iteration, k, n, pi[n,k]))
            for m in range(M):
                fout.write("%d,x,%d,%d,%d,%f\n" % (iteration, k, n, m, x[n,k,m]))
        for m in range(M):
            fout.write("%d,eta,%d,,%d,%f\n" % (iteration, k, m, eta[k,m]))
fout.close()
