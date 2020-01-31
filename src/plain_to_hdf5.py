import h5py
import numpy as np
import argparse

# set up argument parser and parse args
parser = argparse.ArgumentParser(description = \
    'This script converts plain text files to HDF5 format')
parser.add_argument(dest='infile', type=str, \
    help='a path to a plain text data file; one line per observation')
parser.add_argument(dest='outfile', type=str, \
    help='the path and filename of the HDF5 data file to be created')
parser.add_argument('--delim', dest='delim', type=str, default=' ', \
    help='the delimeter to be used in parsing the plain text; default is space')
args = parser.parse_args()

# read in the observed data
obs = np.loadtxt(args.infile, delimiter=args.delim)

# write out data in HDF5 format
f = open(args.outfile, 'w+')
f.close()
f = h5py.File(args.outfile, 'w')
f.create_dataset("observations", data=obs)
f.close()
