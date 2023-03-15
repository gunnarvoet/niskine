# -*- coding: utf-8 -*-
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
cfg = niskine.io.load_config()

# %% [markdown]
# Read Anna's gridded temperature data.

# %%
matfile = cfg.data.gridded.temperature_10m_mat

# %%
tmp = sp.io.loadmat(matfile)

# %%
t = tmp["temperature"]

# %%
t.shape

# %%
tmp["time"]

# %%
time = gv.time.mtlb2datetime(tmp["time"][0])

# %%
time.shape

# %%
tmp.keys()

# %%
tmp["pressure"].shape

# %%
tmp["readme"]

# %% [markdown]
# Okay, it is a bit unclear to me how the depth bins are referenced? And why is pressure gridded on a depth grid?

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True)
gv.plot.axstyle(ax)
ax.pcolormesh(tmp["pressure"][:, :2000])

# %%
fig, ax = gv.plot.quickfig()
ax.pcolormesh(tmp["pressure"][:, :2000])

# %%
file = cfg.data.gridded.temperature_mat

# %%
m = sp.io.loadmat(file)

# %%
m.keys()

# %%
m["readme"]

# %% [markdown]
# The first row of pressure is all nan's, let's throw this out.

# %%
mt = m["temperature"][1:, :]

# %%
mtime = gv.time.mtlb2datetime(m["time"][0])

# %%
mpress = m["pressure"][1:, :]

# %%
dmp = xr.DataArray(mpress.transpose(), coords=dict(time=mtime, zi=np.arange(len(mpress[:, 0]))))

# %%
dmp.sel(time="2019-11").isel(zi=range(40)).diff(dim="zi").gv.tplot(vmin=-10, vmax=10, cmap="RdBu")

# %%
mt.shape

# %%
mpress.shape

# %%
mtime.shape

# %% [markdown]
# Okay, we can now interpolate temperature to a fixed depth grid.

# %%
lon, lat, bottom_depth = niskine.io.mooring_location(mooring=1)

# %% [markdown]
# Let's work in depth coordinates.

# %%
zz = gsw.z_from_p(mpress, lat)

# %%
zz[:, 10000]

# %%
fig, ax = gv.plot.quickfig()
ax.pcolormesh(mt)

# %%
test = mt[1:, 100000]
z = zz[1:, 10000]

# %%
fig, ax = gv.plot.quickfig()
ax.plot(test,z)

# %%
znew = np.arange(-1500, 10, 10)

# %%
from scipy.interpolate import interp1d

# %%
f = interp1d( z[~np.isnan(test)], test[~np.isnan(test)], bounds_error=False)

# %%
fig, ax = gv.plot.quickfig()
ax.plot(test,z)
ax.plot(f(znew), znew)

# %%
zi, ti = mt.shape

# %%
ti

# %%
tnew = np.ones((len(znew), ti)) * np.nan

# %%
tnew.shape

# %%
for i in range(ti):
    mask = ~np.isnan(mt[:, i])
    tnew[:, i] = interp1d( zz[mask, i], mt[mask, i], bounds_error=False)(znew)

# %%
fig, ax = gv.plot.quickfig()
ax.pcolormesh(znew, mtime[:10000], tnew[:, :10000].transpose())

# %%
daa = xr.DataArray(tnew, coords=dict(depth=-znew, time= mtime))

# %% [markdown]
# Pick times where without mooring at surface.

# %%
daa.sel(time="2019-05").gv.tplot()

# %%
daa.sel(time="2020-10").gv.tplot()

# %%
ts = slice("2019-05-19", "2020-10-04")

# %%
da = daa.sel(time=ts).resample(time="1H").mean()

# %%
da.sel(time="2019-11").gv.tplot()

# %%
da.gv.tcoarsen().gv.tplot()

# %%
da.sel(time="2019-11").differentiate(coord="depth").gv.tplot(cmap="RdBu_r", robust=True)

# %% [markdown]
# At least the long term mean shows stable stratification aka a positive temperature gradient towards the ocean surface.

# %%
fig, ax = gv.plot.quickfig(grid=True, w=3.5)
da.mean(dim="time").plot(ax=ax, marker='.', y="depth", yincrease=False)

# %%
fig, ax = gv.plot.quickfig(grid=True, w=3.5)
(-da.mean(dim="time").differentiate(coord="depth")).plot(ax=ax, marker='.', y="depth", yincrease=False)

# %% [markdown]
# I am not sure that I trust this product very much yet. We should look back into pressure interpolation onto the thermistors, I wonder if Anna still has the code, or maybe we should just redo this. There should be enough pressure information to do a decent depth mapping without this many overturns in the data.

# %% [markdown]
# I will save the dataset for now to make some progress with the low mode flux calculation, but this could mess up the estimate for $\eta$ quite a bit. Luckily we are integrating in depth and looking at bigger modes, so maybe not that important, but still not very satisfying.

# %%
da.to_netcdf(cfg.data.gridded.temperature_10m_nc)

# %%
ax = da.gv.tplot(cmap='Spectral_r', cbar_kwargs=dict(label="temperature [$^{\circ}$C]"))
ax.set(title="NISKINe M1 gridded thermistor data")
niskine.io.png("gridded_thermistors")

# %%
