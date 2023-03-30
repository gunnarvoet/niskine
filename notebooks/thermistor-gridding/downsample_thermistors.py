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
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import pandas as pd
import gsw

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
t_2min_dir = cfg.data.proc.thermistor2min
print(t_2min_dir)
t_2min_dir.mkdir(exist_ok=True)

# %%
lon, lat, depth = niskine.io.mooring_location(mooring=1)

# %% [markdown]
# # Downsample NISKINe thermistors

# %% [markdown]
# Downsample and save NISKINe thermistors for gridding.

# %% [markdown]
# ### Time Vector

# %% [markdown]
# Keeping this here from thermistor gridding just in case.

# %%
timeslice = niskine.io.mooring_start_end_time(mooring=1)
print(timeslice.start, "  --  ", timeslice.stop)

# %%
common_time = np.arange(timeslice.start, timeslice.stop, dtype="datetime64[20m]").astype(
    "datetime64[m]"
)

# %% [markdown]
# 2 minute

# %%
common_time_2m = np.arange(timeslice.start, timeslice.stop, dtype="datetime64[2m]").astype(
    "datetime64[m]"
)

# %% [markdown]
# 1 minute

# %%
common_time_1m = np.arange(timeslice.start, timeslice.stop, dtype="datetime64[1m]").astype(
    "datetime64[m]"
)

# %% [markdown]
# ### M1 sensor configuration

# %% [markdown]
# The `.csv` file was exported from the [mooring config spreadsheet](https://docs.google.com/spreadsheets/d/1MQlw1ow0Y2pQBhNj85RbAa9ELnzdttzBYEEfIe2yoRk/edit#gid=2019776936) and sligthly cleaned up afterwards.

# %%
mm = niskine.io.read_m1_sensor_config()
# drop the chipod that has bad pressure
mm = mm.drop(index=614)

# %% [markdown]
# Extract info for all thermistors

# %%
# serial numbers
tsn = mm.where(mm.sensor=="t").dropna(how="all").index.to_numpy()
# full info
tmm = mm.where(mm.sensor=="t").dropna(how="all")

# %%
tsn

# %% [markdown]
# List all thermistors that may have moved from their initial deployment position.

# %%
mm.where(~np.isnan(mm["possible depth offset"])).dropna()

# %%
tsn_trouble = mm.where(~np.isnan(mm["possible depth offset"])).dropna().index.to_numpy()

# %%
tsn_trouble

# %% [markdown]
# We have pressure every 20 min and thus the gridded temperature product can't be on a super fine scale. Maybe we can downsample to every minute? Something that we will need to do though is interpolating to a common time vecctor for all thermistors. Subsample to higher or lower resolution than that?

# %% [markdown]
# Maybe resample (via averaging) to 1 minute resolution and then interpolate to 2 min resolution? We could also low-pass filter instead of calcluating the moving average, this might be faster and less error-prone.

# %% [markdown]
# Maybe simply start out with brute interpolation from the raw data. We can still adjust this later. It would be good to quantify how much of a difference this makes.

# %% [markdown]
# Actually, interpolation does not work well as some of the SBE56 have gaps. Now working with `xarray.resample` to downsample to (defined) 2 minute intervals.

# %%
sn = 425
t = niskine.io.load_thermistor(sn=sn)
t.close()
# t = niskine.io.load_thermistor(sn=sn).interp(time=common_time)

# %%
def interpolate_and_save(sn):
    t = niskine.io.load_thermistor(sn=sn)
    t.close()
    if sn in tsn_trouble:
        intptime = common_time_2m[common_time_2m<np.datetime64("2019-11-12")]
    else:
        intptime = common_time_2m
    tmp = t.interp(time=intptime)
    savename = t_2min_dir.joinpath(f"{sn:06d}.nc")
    tmp.to_netcdf(savename)


# %%
# interpolate_and_save(sn)

# %%
def downsample_and_save(sn):
    print(sn)
    t = niskine.io.load_thermistor(sn=sn)
    t.close()
    tt = t.resample(time="120s", origin=common_time_2m[0]).mean()
    if sn in tsn_trouble:
        tt = tt.where(tt.time<np.datetime64("2019-11-12"), drop=True)
    savename = t_2min_dir.joinpath(f"{sn:06d}.nc")
    tt.to_netcdf(savename)


# %%
for sn in tsn[15:]:
     downsample_and_save(sn)

# %%

# %%

# %% [markdown] heading_collapsed=true
# # dev

# %% hidden=true
tt = t.resample(time="1H", origin=common_time_1m[0]).mean()

# %% hidden=true
common_time_1m

# %% hidden=true
tt.time

# %% hidden=true

# %% hidden=true

# %% hidden=true
tt = t.resample(time="60s").mean()

# %% hidden=true
tt = t.resample(time="60s").mean()
test2 = tt.interp(time=common_time_2m)

# %% hidden=true
(test-test2).plot()

# %% hidden=true
test.sel(time="2019-08-01").plot()
test2.sel(time="2019-08-01").plot()
