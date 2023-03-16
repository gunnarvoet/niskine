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

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "80aa11a68a82c8", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
# %matplotlib inline
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import gvpy as gv
import velosearaptor as trex
import niskine

# %reload_ext autoreload
# %autoreload 2
# %autosave 0

# %config InlineBackend.figure_format = 'retina'

# %%
conf = niskine.io.load_config()

# %% [markdown]
# Link ADCP files into data dir

# %%
mooringdir = Path('/Users/gunnar/Projects/niskine/data/NISKINe/Moorings/NISKINE19')
niskine.io.link_proc_adcp(mooringdir)

# %% [markdown]
# # Shallow ADCP Winter Day Time Gaps

# %% [markdown]
# The shallow ADCPs 3109 and 3110 show very short range during day time in the winter months. Investigate this a bit further and find out if data drop out due to a bad setting or due to instrument limitation.

# %%
a = niskine.io.load_adcp(mooring=1, sn=3109)

# %%
t1 = slice('2019-07-01', '2019-07-31')
t2 = slice('2020-01-09', '2020-01-13')

# %%
a.sel(time=t1).u.dropna(dim='z', how='all').gv.tplot()

# %%
a.sel(time=t1).amp.dropna(dim='z', how='all').gv.tplot()

# %%
a

# %%
a.sel(time=t2).u.dropna(dim='z', how='all').gv.tplot()

# %%
a.sel(time=t2).amp.dropna(dim='z', how='all').gv.tplot()

# %%
a.sel(time=t2).pg.dropna(dim='z', how='all').gv.tplot()

# %% [markdown]
# Load raw data

# %%
rawfile = list(mooringdir.joinpath("M1/ADCP/raw/SN3109/").glob("*.000"))

# %%
rawfile

# %%
rr = trex.io.read_raw_rdi(rawfile[0].as_posix())

# %%
rr

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for i, axi in enumerate(ax):
    rr.sel(time=t2, beam=i+1).amp.dropna(dim='z', how='all').gv.tplot(ax=axi)

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for i, axi in enumerate(ax):
    rr.sel(time=t2, beam=i+1).cor.dropna(dim='z', how='all').gv.tplot(ax=axi)

# %%
