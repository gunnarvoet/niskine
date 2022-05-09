# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
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
conf = niskine.io.load_config()

# %% [markdown]
# Link data files into package data directory. Safe to run if the links already exist.

# %%
mooringdir = '/Users/gunnar/Projects/niskine/data/NISKINe/Moorings/NISKINE19/'
niskine.io.link_proc_adcp(mooringdir=mooringdir)

# %% [markdown]
# ## Create FQ pressure time series

# %%
fq = gv.io.loadmat('/Users/gunnar/Projects/niskine/data/NISKINe/Moorings/NISKINE19/M2/ADCP/proc/FQ10185/fq_converted/FQ_output_EarthCoords.mat')

# %%
fq_datenum = fq['DateNum']
fq_time = gv.time.mtlb2datetime(fq_datenum)

# %%
fq_p_orig = fq['xDucerPresDBar']

# %%
fig, ax = gv.plot.quickfig()
ax.plot(fq_time, fq_p_orig)
gv.plot.concise_date()

# %%
fq_p_orig.mean()

# %% [markdown]
# Load LR above

# %%
lr = niskine.io.load_adcp(mooring=2, sn=10219)
lrp = lr.pressure
lrp.plot()

# %%
lon, lat, depth = niskine.io.mooring_location(mooring=1)

# %%
fqz = -gsw.z_from_p(lrp, lat) + 306

# %% [markdown]
# This is how depth was calculated in the .m file `FQ_interpolateNewPressure.m` (now corrected):

# %%
fqz_a = lrp - lrp.mean() + fq_p_orig.mean()

# %%
fqz.plot()
fqz_a.plot()

# %%
(fqz-fqz_a).mean().item()

# %% [markdown]
# Convert depth back to pressure.

# %%
fq_pressure = gsw.p_from_z(-fqz, lat)

# %% [markdown]
# Interpolate to FQ time vector.

# %%
fq_pressure = fq_pressure.interp(time=fq_time)

# %% [markdown]
# Create empty temperature vector so other routines don't complain that there is none...

# %%
fq_temperature = fq_pressure.copy() * np.nan

# %% [markdown]
# # Convert FQ

# %% [markdown]
# Load .mat file.

# %%
fqmat = gv.io.loadmat(conf.data.proc.adcp.joinpath("FQ_InterpolatedFinal.mat"))

# %%
fig, ax = gv.plot.quickfig()
ax.plot(fqmat['z'][0, :])

# %% [markdown]
# Convert Matlab time.

# %%
fqtime = gv.time.mtlb2datetime(fqmat['dnum'][0, :])

# %% [markdown]
# Interpolate to a regular depth grid.

# %%
znew = np.arange(0, 2016, 16)

unew = np.ones((len(znew), len(fqtime))) * np.nan
vnew = unew.copy()
wnew = unew.copy()

for i, (ui, zi) in enumerate(zip(fqmat['u'].T, fqmat['z'].T)):
    unew[:, i] = sp.interpolate.interp1d(zi, ui, bounds_error=False)(znew)

for i, (vi, zi) in enumerate(zip(fqmat['v'].T, fqmat['z'].T)):
    vnew[:, i] = sp.interpolate.interp1d(zi, vi, bounds_error=False)(znew)

for i, (wi, zi) in enumerate(zip(fqmat['w'].T, fqmat['z'].T)):
    wnew[:, i] = sp.interpolate.interp1d(zi, wi, bounds_error=False)(znew)

# %% [markdown]
# Construct dataset.

# %%
fq = xr.Dataset(
    data_vars=dict(
        u=(("z", "time"), unew),
        v=(("z", "time"), vnew),
        w=(("z", "time"), wnew),
        pressure=(("time"), fq_pressure.data),
        temperature=(("time"), fq_temperature.data),
    ),
    coords=dict(z=(("z"), znew), time=(("time"), fqtime)),
)

# %%
fq.u.attrs = dict(long_name='u', units='m/s')
fq.v.attrs = dict(long_name='u', units='m/s')
fq.w.attrs = dict(long_name='u', units='m/s')
fq.pressure.attrs = dict(long_name='pressure', units='m/s')
fq.temperature.attrs = dict(long_name='temperature', units=r'$^{\circ}$C')
fq.z.attrs = dict(long_name='depth', units='m')

# %%
fq.attrs = dict(project='NISKINe', mooring='M2', sn=10185)

# %%
fq.u.gv.tplot()

# %% [markdown]
# Now save with the same name pattern as the other ADCPs.

# %%
savename = conf.data.proc.adcp.joinpath(f'{fq.attrs["mooring"]}_{fq.attrs["sn"]}.nc')

# %%
fq.to_netcdf(savename)
