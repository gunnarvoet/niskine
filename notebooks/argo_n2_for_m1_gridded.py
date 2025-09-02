# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: python3 (niskine)
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
# # Argo N$^2$

# %%
adcp = niskine.io.load_gridded_adcp(mooring=1)

# %%
m1lon, m1lat, m1depth = niskine.io.mooring_location(mooring=1)
n2a, tz = niskine.clim.climatology_argo_woce(m1lon, m1lat, m1depth)
an2 = niskine.clim.interpolate_seasonal_data(adcp.time, n2a)
n2 = an2.interp_like(adcp)

# %%
# n2a.rename(z="depth").to_netcdf("../data/woce_argo_N2_niskine_M1.nc")

# %%
# n2.to_netcdf("argo_n2_at_m1.nc")
# an2.to_netcdf("argo_n2_at_m1_full_depth.nc")

# %%
an2.isel(time=10).plot(y="z")
n2.isel(time=10).plot(y="z")

# %% [markdown]
# Save mean N$^2$ profile as .mat file.

# %%
n2a.mean(dim="time").plot()

# %%
out = dict(N2=n2a.mean(dim="time").data, depth=n2a.z.data)

# %%
gv.io.savemat(out, "../data/N2.mat")

# %%
test = gv.io.loadmat("../data/N2.mat")

# %%
fig, ax = gv.plot.quickfig()
ax.plot(np.sqrt(test["N2"]), test["depth"])
ax.set(xscale="log")
ax.invert_yaxis()

# %%
