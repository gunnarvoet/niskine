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
import gsw

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
# # NI Filtering

# %% [markdown]
# Apply bandpass filter to ADCP velocity. Mooring knockdown causes some depth levels to have NaN's so we have to deal with those.

# %%
adcp = niskine.io.load_gridded_adcp(mooring=1)

# %%
m1lon, m1lat, m1depth = niskine.io.mooring_location(mooring=1)
n2a, tz = niskine.clim.climatology_argo_woce(m1lon, m1lat, m1depth)
an2 = niskine.clim.interpolate_seasonal_data(adcp.time, n2a)
n2 = an2.interp_like(adcp)

# %%
n2.to_netcdf("argo_n2_at_m1.nc")

# %%
an2.to_netcdf("argo_n2_at_m1_full_depth.nc")

# %%
an2.isel(time=10).plot(y="z")
n2.isel(time=10).plot(y="z")

# %%
