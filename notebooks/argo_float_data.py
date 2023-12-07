# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
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
import gsw
from argopy import DataFetcher as ArgoDataFetcher

import gvpy as gv
import niskine

# %reload_ext autoreload
# %autoreload 2
# %autosave 0

# %config InlineBackend.figure_format = 'retina'

# %%
cfg = niskine.io.load_config()

# %%
mld = xr.open_dataarray(cfg.data.ml.mld)

# %% [markdown]
# # Argo Float Data

# %% [markdown]
# https://argopy.readthedocs.io/en/latest/usage.html

# %%
m1lon, m1lat, m1depth = niskine.io.mooring_location(mooring=1)

# %%
deltalat = 1
deltalon = np.cos(np.deg2rad(m1lat)) / deltalat

# %%
argo1d = ArgoDataFetcher().region([m1lon-deltalon, m1lon+deltalon, m1lat-deltalat, m1lat+deltalat, 0, 2000, '2019-05', '2020-11']).to_xarray()

# %%
argo = argo1d.argo.point2profile()

# %% [markdown]
# Only 10 profiles, none of them in the winter.

# %%
argo.TIME.data
