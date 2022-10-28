# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] heading_collapsed=true
# ##### Imports

# %% hidden=true
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
import gsw
import scipy

import gvpy as gv

import niskine

# %reload_ext autoreload
# %autoreload 2

plt.ion()

# %config InlineBackend.figure_format = 'retina'

# %% hidden=true
conf = niskine.io.load_config()

# %% [markdown]
# # ARGO Mixed Layer Depth Climatology

# %% [markdown]
# Read Argo MLD climatology. MLD is calculated via the density algorithm or the density standard.

# %%
mld_all = xr.open_dataset(conf.data.input.argo_mld)

# %%
mld_da = xr.DataArray(data=mld_all.mld_da_mean.data, coords=dict(lon=mld_all.lon.data, lat=mld_all.lat.data, month=mld_all.month.data), dims=('lat', 'lon', 'month'), name='da')

# %%
mld_dt = xr.DataArray(data=mld_all.mld_dt_mean.data, coords=dict(lon=mld_all.lon.data, lat=mld_all.lat.data, month=mld_all.month.data), dims=('lat', 'lon', 'month'), name='dt')

# %%
mld_argo = xr.merge([mld_dt, mld_da])

# %%
mld_argo.to_netcdf(conf.data.ml.mld_argo)

# %% [markdown]
# ## Compare with NISKINe MLD

# %% [markdown]
# Load NISKINe M1 location

# %%
lon, lat, dep = niskine.io.mooring_location(mooring=1)

# %% [markdown]
# Load MLD from M1

# %%
mldm1 = xr.open_dataarray(conf.data.ml.mld)

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True)
mldm1.groupby("time.month").mean().plot(color='C0', yincrease=False, label='NISKINe M1')
mld_da.interp(lon=lon, lat=lat).plot(color='C4', yincrease=False, label='Argo MLD Climatology (algorithm)')
mld_dt.interp(lon=lon, lat=lat).plot(color='C6', yincrease=False, label='Argo MLD Climatology [threshold]')
ax.legend()
ax.set(ylabel='MLD [m]')
niskine.io.png('mld_argo_and_niskine', subdir='mixed-layer')

# %%

# %%
mld_argo.da.isel(month=8).plot()
