# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python [conda env:niskine]
#     language: python
#     name: conda-env-niskine-py
# ---

# %% [markdown]
# #### Imports

# %%
# # %load /Users/gunnar/Projects/python/standard_imports.py
# %matplotlib inline
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from pathlib import Path
import cartopy.crs as ccrs

import gvpy as gv

import niskine

# %reload_ext autoreload
# %autoreload 2

# %config InlineBackend.figure_format = 'retina'

# %%
cfg = niskine.io.load_config()

# %% [markdown]
# # NISKINe Bathymetry

# %% [markdown]
# Load Smith & Sandwell

# %%
ss = gv.ocean.smith_sandwell(lon=alt.lon, lat=alt.lat)

# %%
ss.plot()

# %% [markdown]
# Load processed multibeam data

# %%
mb = xr.open_dataarray(conf.data.proc.mb)

# %%
mb

# %% [markdown]
# Reduce in size for faster plotting - don't need the great resolution here.

# %%
mbc = mb.coarsen({'lon': 10, 'lat': 10}, boundary='pad')
b = mbc.mean()

# %%
b.plot()
