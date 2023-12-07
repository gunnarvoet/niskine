# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
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
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import pandas as pd
import gsw
from tqdm.notebook import tqdm

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

# %%
lon, lat, depth = niskine.io.mooring_location(mooring=1)

# %% [markdown]
# # Save 2-minute thermistor time series

# %%
m = niskine.strat.MooringConfig()

# %% [markdown]
# Reading the 2min downsampled versions here. They should all be on the same time grid. Don't read the thermistor near the bottom (it won't help much in the gridding process).

# %%
# path to data files
t2mindir = cfg.data.proc.thermistor2min

# %%
allt = []
for sn in m.sn_thermistor:
    allt.append(xr.open_dataarray(t2mindir.joinpath(f"{sn:06d}.nc")))
#     allt[-1]

# %% [markdown]
# All temperature time series start at the same time:

# %%
start_times = [ai.time[0].data for ai in allt]
np.diff(start_times)

# %% [markdown]
# Gather them into one data matrix (`xarray.concat` is super slow at combining the time series as they are of different lengths.

# %%
len_allt = [len(ai) for ai in allt]
ni = np.max(len_allt)
ii = np.argmax(len_allt)
mi = len(allt)
print(mi, ni)

# %%
time = allt[ii].time.copy()
temperature_matrix = np.ones([mi, ni]) * np.nan
for i, ai in enumerate(allt):
    temperature_matrix[i, 0:len(ai.data)] = ai.data

temp = xr.DataArray(temperature_matrix, dims=["depth_nominal", "time"], coords=dict(time=(("time"), time.data), depth_nominal=(("depth_nominal"), m.nomz_thermistor)))
temp.coords["sn"] = (("depth_nominal", m.sn_thermistor))
temp.depth_nominal.attrs = dict(long_name='nominal depth', units='m')
temp.sn.attrs = dict(long_name='serial number', units='')
temp.attrs = dict(long_name='temperature', units='°C')

# %%
temp.sel(time=slice("2019-06-08", "2019-06-15")).gv.tplot(yincrease=False)

# %%
temp.sel(time="2019-12").gv.tplot(yincrease=False)
# ax.set(ylim=[200, 20])

# %%
temp.sel(time=slice("2019-10-05", "2019-10-10")).gv.plot(hue="sn", add_legend=False, color="0.2", linewidth=0.1);

# %% [markdown]
# ### Save 2-minute time series

# %%
cfg.data.gridded.temperature_thermistor_2min

# %%
temp.to_netcdf(cfg.data.gridded.temperature_thermistor_2min)
