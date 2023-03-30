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
lon, lat, depth = niskine.io.mooring_location(mooring=1)

# %% [markdown]
# Link processed thermistor files into package data directory.

# %%
niskine.io.link_proc_temperature(Path("/Users/gunnar/Projects/niskine/data/NISKINe/Moorings/NISKINE19"))

# %% [markdown]
# # Inspect NISKINe M1 thermistors

# %% [markdown]
# Some of the thermistors were found on recovery to have moved from their initial deployment depth. Look at these here.

# %% [markdown]
# Summary: Most questionable thermistors slid towards chipods for which I don't have temperature time series handy. In the one case where the thermistor slid to another thermistor I can see a clear change in the time series of the difference between those two around 12 Nov 2019. We should assume that after this date all thermistors that moved have bad data.

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
tmm = mm.where(mm.sensor=="t").dropna(how="all")

# %%
tsn

# %% [markdown]
# List all thermistors that may have moved from their initial deployment position.

# %%
mm.where(~np.isnan(mm["possible depth offset"])).dropna()

# %% [markdown]
# ### SN6413 vs 6435

# %% [markdown]
# I have notes on the mooring diagram that the electroncis for 6413 and 6435 were swapped. This appears to be wrong - 6413 definitely has colder temperatures than 6435.
#
# I will correct this now on the mooring config and add a note on the scanned logsheet.

# %%
sn = 6413
sna = 6435

# %%
print(mm.loc[sn].depth)
print(mm.loc[sna].depth)

# %%
t = niskine.io.load_thermistor(sn=sn).interp(time=common_time)
ta = niskine.io.load_thermistor(sn=sna).interp(time=common_time)

# %%
print(t.mean().item())
print(ta.mean().item())

# %% [markdown]
# ### SN425

# %% [markdown]
# May have slid up to chipod above. We can compare to temperature measured on 72186 2.5m above the chipod (and nominally 5m above this sensor).

# %%
sn = 425
sna = 72186

# %%
t = niskine.io.load_thermistor(sn=sn).interp(time=common_time)
ta = niskine.io.load_thermistor(sn=sna).interp(time=common_time)

# %%
(t-ta).plot()

# %%
(ta.mean() - t.mean()).item()

# %%
(ta.sel(time="2019").mean() - t.sel(time="2019").mean()).item()

# %%
(ta.sel(time="2020").mean() - t.sel(time="2020").mean()).item()

# %%

# %% [markdown]
# ### SN72202

# %% [markdown]
# May have slid down to thermistor 72196. This one shows an obvious change in Nov 2019.

# %%
sn = 72202
sna = 72196

# %%
t = niskine.io.load_thermistor(sn=sn).interp(time=common_time)
ta = niskine.io.load_thermistor(sn=sna).interp(time=common_time)

# %%
(t-ta).plot()

# %%
(t-ta).sel(time="2019-11").plot()

# %%
(t.mean() - ta.mean()).item()

# %%
(t.sel(time="2019").mean() - ta.sel(time="2019").mean()).item()

# %%
(t.sel(time="2020").mean() - ta.sel(time="2020").mean()).item()

# %% [markdown]
# ### SN72158

# %% [markdown]
# May have slid up to microcat 2864. The comparison may also show reduced variance starting in Nov 2019.

# %%
sn = 72158
sna = 2864

# %%
t = niskine.io.load_thermistor(sn=sn).interp(time=common_time)
m = niskine.io.read_microcats()
ta = m.sel(sn=sna).t.interp(time=common_time)

# %%
(t-ta).plot()

# %%
(ta.mean() - t.mean()).item()

# %%
(ta.sel(time="2019").mean() - t.sel(time="2019").mean()).item()

# %%
(ta.sel(time="2020").mean() - t.sel(time="2020").mean()).item()

# %% [markdown]
# ### SN5462

# %% [markdown]
# May have slid up to chipod above. We can compare to temperature measured on 72147 15m below (probably too far away to see anything).

# %%
sn = 5462
sna = 72147

# %%
t = niskine.io.load_thermistor(sn=sn).interp(time=common_time)
ta = niskine.io.load_thermistor(sn=sna).interp(time=common_time)

# %%
(t-ta).plot()

# %%
(t.mean() - ta.mean()).item()

# %%
(t.sel(time="2019").mean() - ta.sel(time="2019").mean()).item()

# %%
(t.sel(time="2020").mean() - ta.sel(time="2020").mean()).item()

# %% [markdown]
# ### SN6413

# %% [markdown]
# May have slid up to chipod above. We can compare to temperature measured on 72147 2.5m above the chipod.

# %%
sn = 6413
sna = 72147

# %%
t = niskine.io.load_thermistor(sn=sn).interp(time=common_time)
ta = niskine.io.load_thermistor(sn=sna).interp(time=common_time)

# %%
(t-ta).plot()

# %%
(ta.mean() - t.mean()).item()

# %%
(ta.sel(time="2019").mean() - t.sel(time="2019").mean()).item()

# %%
(ta.sel(time="2020").mean() - t.sel(time="2020").mean()).item()

# %% [markdown]
# ### SN6420 vs 6447

# %% [markdown]
# Nothing slipped here but the spreadsheet has 6447 shallower than 6420 which can't be true as it was on the float below the mooring line that 6420 was mounted on. Compare mean temperatures here.
#
# 6447 is colder on average than 6420 so that makes sense then. I will correct the spreadsheet.

# %%
sn = 6420
sna = 6447

# %%
print(mm.loc[sn].depth)
print(mm.loc[sna].depth)

# %%
t = niskine.io.load_thermistor(sn=sn).interp(time=common_time)
ta = niskine.io.load_thermistor(sn=sna).interp(time=common_time)

# %%
print(t.mean().item())
print(ta.mean().item())

# %% [markdown]
# ### Mean depth of deep microcat

# %%
m = niskine.io.read_microcats()
mm = m.sel(sn=12712)

# %%
mm.p.plot()

# %%
gsw.z_from_p(2891, lat)

# %% [markdown]
# This is a bit different from the 2850m nominal depth but could be due to a) uncertainty in the pressure measurement and b) uncertainty in the conversion from p to z which really is based on some average stratification at this latitude. The conversion higher up in the water column wouldn't be as affected as pressure from this deep sensor.
