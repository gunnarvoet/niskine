# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Imports

# %%
# %matplotlib inline
from pathlib import Path
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import gsw
import cmocean

import gvpy as gv

import niskine

# %config InlineBackend.figure_format = 'retina'

# %reload_ext autoreload
# %autoreload 2
# %autosave 300


# %% [markdown]
# Load configuration from `config.yml` in the root directory. `io.load_config()` automatically detects the root directory and adjusts the paths.

# %%
config = niskine.io.load_config()

# %% [markdown]
# Print the config.

# %%
niskine.io.print_config(print_values=True)

# %% [markdown]
# # M1

# %% [markdown]
# After copying or linking the processed M1 ADCP netcdf files into `data/proc/adcp` we can list and then read them.

# %%
m1_files = sorted(config.data.proc.adcp.glob('*.nc'))

# %% [markdown]
# Let's look at only one ADCP file for now.

# %%
adcp = xr.open_dataset(m1_files[0])

# %%
adcp

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True)
adcp.u.plot(ax=ax)
ax.set(xlabel='')
ax.invert_yaxis()

# %% [markdown]
# Extract numpy arrays and make a plot without xarray.

# %%
u = adcp.u.data
time = adcp.time.data
z = adcp.z.data

# %%
u.shape

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True)
h = ax.pcolormesh(time, z, u, cmap='RdBu_r', vmin=-0.6, vmax=0.6)
plt.colorbar(h)
ax.set(ylim=(1700, 800))
