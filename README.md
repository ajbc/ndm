# Nonparmetric Deconvolution Models (NDMs)
(C) Copyright 2016-2020, Allison J.B. Chaney and Archit Verma

This software is distributed under the MIT license.  See `LICENSE.txt` for details.

#### Repository Contents
- `dat` example data
- `doc` paper LaTeX
- `src` python source code

## Dependencies
We recommend starting with a fresh [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and running the following to install the dependencies for this project. 

```
conda install h5py 
pip install scipy sklearn-extensions matplotlib
```

## Data
This NDM code reads in data from an [HDF5](https://support.hdfgroup.org/HDF5/whatishdf5.html) file.
We provide `dat/voting/cal-precinct-prop-data.hdf5` as an example data set (obtained from [here]{https://github.com/datadesk/california-2016-election-precinct-maps) and described in detail [here](https://www.latimes.com/projects/la-pol-ca-california-neighborhood-election-results/]).


#### Converting Plain Text
Since most real-world data are not stored in the HDF5 format, we provide the script `src/plain_to_hdf5.py` to help convert plain text files into the required format.
It is used as follows.
```
python src/plain_to_hdf5.py infile outfile [--delim DELIM]
```

More concretely, imagine we have a comma-separated file `dat/voting/cal-precinct-prop-data.csv`.  To convert this, we run the following:
```
python src/plain_to_hdf5.py dat/voting/cal-precinct-prop-data.csv dat/voting/cal-precinct-prop-data.hdf5 --delim ','
```

#### Simulating Data
As an alternative to real-world data, we can use `src/simulate.py` to simuate data.  The arguments for this script are shown below.  In the output directory, it will creat two files: `settings.dat` and `data.hdf5`.  The former is a log of all the settings used to generate the data.  The latter is contains the simulated data in the format required by the NDM code as well as the "ground truth" intermediate parameters used to generate the data.

|Option|Arguments|Help|Default|
|---|---|---|---|
|help||show help message||
|out|OUTDIR|output directory||
|msg|MESSAGE|specify a log message||
|seed|SEED|random seed|from time|
|K|K|number of global latent factors|10|
|N|N|number of observations|100|
|M|M|number of obsevered features|10|
|domain|DOMAIN|the domain of obsewrvations; one of: real, unit, positive, integer|real|
|proc|PROCEDURE|which simulation procedure to use (1 through 5)|1|

Example: `python src/simulate.py --out dat/sim_example`

## Fitting an NDM
To fit an NDM, use the `main.py` script; for example, with the simulated data we just created, we could run the following.
```
python src/main.py --data dat/sim_example/data.hdf5
```

#### Options
|Option|Arguments|Help|Default|
|---|---|---|---|
|help||show help message||
|data|DATA|a path to an HDF5 data file||
|out|OUTDIR|output directory|out|
|msg|MESSAGE|specify a log message||
|save_freq|SAVE_FREQ|frequency of saving current status|10|
|save_all||do not overwrite intermediate saved states|overwrite|
|conv_thresh|CONV_THRESH|convergence threshold to stop fit|1e-4|
|min_iter|MIN_ITER|minimum number of iterations|25|
|max_iter|MAX_ITER|maximum number of iterations|1000|
|batch_max_iter|BATCH_MAX_ITER|maximum number of iterations per batch (nonparametric only)|20|
|tau|TAU|stochastic gradient delay|1024|
|kappa|KAPPA|stochastic gradient forgetting rate|0.7|
|sample_size|SAMPLE_SIZE|stochastic gradient sample size|64|
|seed|SEED|random seed|from time|
|cores|CORES|number of cores to use|1|
|K|K|initial (if nonparametric) or fixed K|10|
|fix_K||flag to fix the number of latent components K|off (nonparametric)|
|gbl_con|GBL_ALPHA|global concentration parameter|1.0|
|lcl_con|LCL_ALPHA|local concentration parameter|10.0|
|rho|RHO|local counts prior|1.0|
|dist|F_DIST|distribution f for observtions y; options: normal, log_normal, gamma, exponential, poisson, beta|normal|
|link|G_LINK|link function for observations y; options: identity (default for normal, log_normal), exp, softplus (default for gamma, poisson), sigmoid (default for beta), expinverse (default for exponential)|depends on f distribution|

## Exploring Model Results
To process multiple iterations of model fit parameters, use `collapse_model_results.py` in the `src` directory.  This script takes ones argument: the fit output directory.  It produces a single file named `collapsed_model_results.csv` which contains the fitted parameter values by iteration in long format.  This is primarily for use with very small datasets.
To similarly process simulation parameters into a long format for exploration, use `reformat_simulation_parameters.py`, which takes one argument (the original HDF5 file); this script produces a file named `simulation_parameters.csv` in the same directory as the source HDF5 file.
