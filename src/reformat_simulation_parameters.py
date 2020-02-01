import os, sys, h5py
import numpy as np

if len(sys.argv) != 2:
    print("usage: python reformat_simulation_parameters.py [hdf5 file]")
    sys.exit(-1)

filename = sys.argv[1]
fout = open(os.path.join(os.path.dirname(filename), \
    'simulation_parameters.csv'), 'w+')
fout.write("parameter,K,N,M,value\n")

print("processing", filename, "...")
f = h5py.File(filename, 'r')

beta = np.array(f['global_concentration'])
pi = np.array(f['local_concentration'])
eta = np.array(f['global_features'])
x = np.array(f['local_features'])
y = np.array(f['observations'])

K = beta.shape[0]
N = pi.shape[0]
M = eta.shape[1]

for k in range(K):
    fout.write("beta,%d,,,%f\n" % (k, beta[k]))
    for n in range(N):
        fout.write("pi,%d,%d,,%f\n" % (k, n, pi[n,k]))
        for m in range(M):
            fout.write("x,%d,%d,%d,%f\n" % (k, n, m, x[n,k,m]))
    for m in range(M):
        fout.write("eta,%d,,%d,%f\n" % (k, m, eta[k,m]))
for n in range(N):
    for m in range(M):
        fout.write("y,,%d,%d,%f\n" % (n, m, y[n,m]))

fout.close()
