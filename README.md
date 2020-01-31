# Nonparmetric Deconvolution Models (NDMs)
(C) Copyright 2016-2018, Allison J.B. Chaney and Archit Verma

This software is distributed under the MIT license.  See `LICENSE.txt` for details.

#### Repository Contents
- `dat` example data
- `doc` JMLR paper LaTeX
- `src` python source code

## Data
This NDMs code reads in data from an [HDF5](https://support.hdfgroup.org/HDF5/whatishdf5.html) file.
We provide `dat/GSE11058_trans.hdf5` as an example data set (obtained from (here)[http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE11058] and described in detail (here)[http://www.ncbi.nlm.nih.gov/pubmed/19568420]).  (TODO: add src/description of this data; and/or add a bunch of data and list it all here.)


#### Converting Plain Text
Since most real-world data are not stored in the HDF5 format, we provide the script `src/plain_to_hdf5.py` to convert plain text files into the required format.
It is used as follows.
```
python src/plain_to_hdf5.py infile outfile [--delim DELIM]
```

More concretely, imagine we have a comma-separated file `dat/GSE11058_trans.csv`.  To convert this, we run the following:
```
python src/plain_to_hdf5.py dat/GSE11058_trans.csv dat/GSE11058_trans.hdf5 --delim ','
```

#### Simulating Data
As an alternative to real-world data, we can use `src/simulate.py` to simuate data.  The arguments for this script are shown below.  In the output directory, it will creat two files: `settings.dat` and `data.hdf5`.  The former is a log of all the settings used to generate the data.  The latter is contains the simulated data in the format required by the NDM code as well as the "ground truth" intermediate parameters used to generate the data.

|Option|Arguments|Help|Default|
|---|---|---|---|
|help||show help message||
|out|OUTDIR|output directory||
|msg|MESSAGE|specify a log message||
|seed|SEED|random seed|from time|
|K|K|number of global latent components|10|
|N|N|number of observations|100|
|P|P|number of features|10|
|gbl_con|GBL_ALPHA|global concentration parameter|10.0|
|lcl_con|LCL_ALPHA|local concentration parameter|1.0|
|gbl_dist|BASE_DIST|base distribution for global factors, options: normal, multivariate_normal, log_normal, multivariate_log_normal, exponential, gamma, poisson|normal|
|lcl_dist|BASE_DIST|distribution for local factors, options: normal, multivariate_normal, log_normal, multivariate_log_normal, exponential, gamma, poisson|normal|
|glm_dist|GLM_DIST|distribution for generalized linear model at local level, options: normal, multivariate_normal, log_normal, multivariate_log_normal, exponential, gamma, poisson|normal|
|glm_link|GLM_LINK|link function for generalized linear model at local level, options: identity, inverse, inversesquared, log, logit|identity|


## Running NDMs
TODO

#### Options
|Option|Arguments|Help|Default|
|---|---|---|---|
|help||show help message||
|data|DATA|a path to an HDF5 data file||
|out|OUTDIR|output directory|out|
|msg|MESSAGE|specify a log message||
|save_freq|SAVE_FREQ|frequency of saving current status|10|
|save_all||do not overwrite intermediate saved states |overwrite|
|conv_thresh|CONV_THRESH|convergence threshold to stop fit|1e-4|
|min_iter|MIN_ITER|minimum number of iterations|40|
|max_iter|MAX_ITER|maximum number of iterations|1000|
|tau|TAU|stochastic gradient delay|1024|
|kappa|KAPPA|stochastic gradient forgetting rate|0.7|
|sample_size|SAMPLE_SIZE|stochastic gradient sample size|64|
|seed|SEED|random seed|from time|
|gbl_con|GBL_ALPHA|global concentration parameter|1.0|
|lcl_con|LCL_ALPHA|local concentration parameter|1.0|
|gbl_dist|BASE_DIST|base distribution for global factors, options: normal, multivariate_normal, log_normal, mulitvariate_log_normal|normal|
|lcl_dist|LCL_DIST|distribution for local factors, options: normal, multivariate_normal, log_normal, mulitvariate_log_normal|normal|
|lcl_dist|GLM_DIST|distribution for generalized linear model at local level, options: normal, log_normal|normal|
|lcl_link|GLM_LINK|link function for generalized linear model at local level, options: identity, inverse, inversesquared, log, logit|identity|

## Exploring Model Results
TODO: describe visualization/exploration process

To process multiple iterations of model fit parameters, use `collapse_model_results.py` in the `src` directory.  This script takes ones argument: the fit output directory.  It produces a single file named `collapsed_model_results.csv` which contains the fitted parameter values by iteration in long format.  This is primarily for use with very small datasets.
To similarly process simulation parameters into a long format for exploration, use `reformat_simulation_parameters.py`, which takes one argument (the original HDF5 file); this script produces a file named `simulation_parameters.csv` in the same directory as the source HDF5 file.
The script `explore_parameters.R` can be used on these files to produce exploratory figures, as follows.
```
Rscript explore_parameters.R [collapsed model results] [optional: simulated parameters]
```


## Running an Experiment
TODO: describe our evaluation/comparison model pipleine


- Install dependencies for comparison methods: [Scikit-learn](http://scikit-learn.org/stable/install.html), [GSL](https://www.gnu.org/software/gsl/), [Armadillo](http://arma.sourceforge.net/download.html), [PyMC3](https://pymc-devs.github.io/pymc3/notebooks/getting_started.html#Installation), [Pandas](http://pandas.pydata.org/pandas-docs/stable/install.html)
- From the `src` directory, run `./setup_experiments.sh` to download data and source code for comparison methods.
- Also from the `src` directory, run `./run_experiments.sh K` with your choice of `K` to run a set of experiments.


Current comparison methods:
- kmeans.py
- gap.py (gamma-poisson factorization)
- pmf.py (probablistic matrix factorization)
- nmf.py (non-negative matrix facorization)
