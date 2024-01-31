#!/usr/bin/env python
# coding: utf-8

# ## Example with HCV data

# In[1]:


# Formatting and plotting packages
# %load_ext nb_black
import matplotlib.pyplot as plt

# from IPython.display import set_matplotlib_formats

# get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['svg']")

from Bio import AlignIO
import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.special import xlogy

jax.config.update("jax_enable_x64", True)

import sys
sys.path.append("/Users/ys/Desktop/Research/vbcompare/")

from vbsky.fasta import SeqData
from vbsky.bdsky import _lognorm_logpdf
from vbsky.prob import VF
from vbsky.prob.transform import (
    Transform,
    Compose,
    Positive,
    ZeroOne,
    DiagonalAffine,
    Exp,
)
from vbsky.prob.distribution import Constant
from vbsky.plot import plot_helper, plot_one

# add this to make sure multiprocessing works in command line
import multiprocessing
multiprocessing.set_start_method('fork')

# set number of tips
import argparse

parser = argparse.ArgumentParser(description='BDSKY.')
parser.add_argument('--ntips', type=int,
                    help='an integer for the number of tips')
parser.add_argument('--epochs', type=int, default=10,
                    help='an integer for the number of epochs')
parser.add_argument('--niter', type=int, default=20,
                    help='an integer for the number of iterations')

args = parser.parse_args()

# #### Import Data

# In[2]:


hcv = AlignIO.read(f"existing_{args.ntips}.nexus", format="nexus")
data = SeqData(hcv, gisaid=False, contemp=True, dates=1993.0)


# #### Data Preprocessing
# 
# In this step we are 
# 1. Sampling with replacement to create subsamples 
# 2. Estimating the tree topologies for each subsample
# 3. Preprocessing data to be used in the optimizer

# In[3]:


n_tips = args.ntips  # Number of tips per tree
n_trees = 1  # Number of trees
temp_folder = "./"  # Place to save temporary folders
# tree_path = "./subsample.trees"

existing_tree_path = f"./{args.ntips}.tre"

# data.prep_data(
#     n_tips,
#     n_trees,
#     temp_folder,
#     tree_path,
# )

data.prep_data_with_existing_tree(existing_tree_path)


# #### Define Parameters of the Model
# 
# We define parameters through "flows" which are simply a series of transformations. Recall that we use a mean field approximation, so the posteriors are normal distributions. By transforming the normal distributions with the flows we can ensure that various parameters are constrained to the correct set of values.
# 
# Below we define two flows that we will use many times: 'pos' ensures the parameter is positive and 'z1' ensures the parameter is in (0,1). There are many more transformations available and longer sequences of transformations can also be chained for more flexible posterior distributions. If we want to fix parameters we can simply set them to be constant.

# In[4]:


pos = Compose(DiagonalAffine, Exp)
z1 = Compose(DiagonalAffine, ZeroOne)


# In[5]:


m = args.epochs  # Set the number of intervals (length of vector of BDSKY parameters)

# Define local parameters for the different trees in our ensemble using z1 flow to ensure branch parameters
# are constrained appropriately
# local_flows = [
#     {"proportions": Transform(td.n - 2, z1), "root_proportion": Transform(1, z1)}
#     for td in data.tds
# ]
local_flows = [
    {"proportions": Constant(np.ones(td.n - 2, )), "root_proportion": Constant(1.0)}
    for td in data.tds
]

# Since we only have samples from a single time point we can set the sampling rate $s$ and origin_start to be fixed values. We also fix the clock rate. For the other variables we use either the zero-one or exponential transformation to constrain appropriately.

# In[6]:


# Define global epidemiological parameters
# global_flows = VF(
#     origin=Transform(1, pos),
#     origin_start=Constant(0.0),
#     delta=Transform(m, pos),
#     R=Transform(m, pos),
#     rho_m=Transform(1, z1),
#     s=Constant(np.repeat(0.00, m)),
#     precision_R=Transform(1, pos),
#     precision_delta=Transform(1, pos),
#     clock_rate=Constant(0.79e-3),
# )

global_flows = VF(
    origin=Constant(0.0),
    origin_start=Constant(0.0),
    delta=Transform(m, pos),
    R=Transform(m, pos),
    rho_m=Transform(1, z1),
    s=Transform(m, z1), # Constant(np.repeat(0.00, m)),
    precision_R=Constant(1.0), # Transform(1, pos),
    precision_delta=Constant(1.0), # Transform(1, pos),
    clock_rate=Constant(0.79e-3),
)


data.setup_flows(global_flows, local_flows)


# #### Define Priors
# 
# To match the flexibility of the flows, users must provide the prior distributions. We need to provide priors for all  parameters defined to be non-constant. Additionally, we can add a GMRF smoothing parameter to enforce smoother estimates of the BDSKY parameters. 

# In[7]:


# Define a function to give value of prior given parameters
def _params_prior_loglik(params):
    ll = 0
    # Hyperprior for gmrf smoothing prior
    tau = {"R": params["precision_R"][0], "delta": params["precision_delta"][0]}
    ll += jax.scipy.stats.gamma.logpdf(tau["R"], a=0.001, scale=1 / 0.001)
    ll += jax.scipy.stats.gamma.logpdf(tau["delta"], a=0.001, scale=1 / 0.001)
    ll += jax.scipy.stats.beta.logpdf(params["rho_m"], 1, 9999).sum()

    mus = [1.0, 1.0, 1.0]
    sigmas = [1.25, 1.25, 1.25]

    for i, k in enumerate(["R", "delta", "origin"]):
        log_rate = jnp.log(params[k])
        ll += _lognorm_logpdf(log_rate, mu=mus[i], sigma=sigmas[i]).sum()

    # gmrf smoothing
    for k in ["R", "delta"]:
        log_rate = jnp.log(params[k])
        ll -= (tau[k] / 2) * (jnp.diff(log_rate) ** 2).sum()
        m = len(log_rate)
        ll += xlogy((m - 1) / 2, tau[k] / (2 * jnp.pi))
    # print("Shaoyucai: Prior!")
    return ll


# #### Run Optimization Loop
# 
# After we have defined our parameters we can run the optimization step and sample from the posterior distributions.

# In[8]:


rng = jax.random.PRNGKey(6)
n_iter = args.niter
threshold = 0.001
step_size = 1.0

res = data.loop(
    _params_prior_loglik,
    rng,
    n_tips,
    n_epochs=m,
    n_iter=n_iter,
    step_size=step_size,
    threshold=threshold,
    dbg=False
)


# #### Visualize Posteriors

# In[9]:


fig, ax = plt.subplots()
start, top, end, x0 = plot_helper(res, data, n_tips)
plot_one(res, ax, "R", m, start, top, end, x0, "R", "fill", "")
plot_one(res, ax, "delta", m, start, top, end, x0, "delta", "fill", "")
ax.legend()

