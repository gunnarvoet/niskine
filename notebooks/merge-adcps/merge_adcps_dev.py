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

# %% [markdown] heading_collapsed=true
# ### Imports

# %% hidden=true
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


# %% [markdown] hidden=true
# Load configuration from `config.yml` in the root directory. `io.load_config()` automatically detects the root directory and adjusts the paths.

# %% hidden=true
conf = niskine.io.load_config()

# %% [markdown] hidden=true
# Link data files into package data directory. Save to run if the links already exist.

# %% hidden=true
mooringdir = '/Users/gunnar/Projects/niskine/data/NISKINe/Moorings/NISKINE19/'
niskine.io.link_proc_adcp(mooringdir=mooringdir)

# %% [markdown] hidden=true
# After copying or linking the processed M1 ADCP netcdf files into `data/proc/adcp` we can list and then read them.

# %% hidden=true
# m1_files = sorted(conf.data.proc.adcp.glob('M1*.nc'))

# %% [markdown]
# # Merge NISKINe ADCPs

# %% [markdown]
# This notebook illustrates a few of the steps involved in merging the ADCP time series per mooring. The actual merging now happens in [merge_adcps](merge_adcps.ipynb).

# %% [markdown] heading_collapsed=true
# ### Read all ADCPs

# %% hidden=true
aa = niskine.merge.load_mooring_adcps(mooring=1)

# %% [markdown] heading_collapsed=true
# ### Select and sort

# %% [markdown] hidden=true
# Select the ones that lasted into 2020.

# %% hidden=true
asel = niskine.merge.select_adcps(aa, '2020')
print(len(asel))

# %% [markdown] hidden=true
# Okay, let's cut down to time at depth. I determined the start and end times for each mooring from the pressure time series and put them into `niskine.io.mooring_start_end_time()`.

# %% hidden=true
a = niskine.merge.at_depth_only(asel, mooring=1)

# %% hidden=true
fig, ax = gv.plot.quickfig()
[ai.pressure.gv.tplot(ax=ax) for ai in a]
ax.invert_yaxis()

# %% [markdown] hidden=true
# Sort the ADCPs by their mean pressure record.

# %% hidden=true
asort = niskine.merge.sort_in_depth(a)

# %% [markdown] heading_collapsed=true
# ### Interpolate time

# %% [markdown] hidden=true
# Find time and depth interval for each ADCP. Interpolate to a common time vector. Interpolate to a common depth vector (with the same spacing for all depths?).

# %% hidden=true
niskine.merge.print_sampling_period(asort)

# %% [markdown] hidden=true
# What is a good time vector to interpolate to? Just do hourly for now even though this kind of sucks for the slower sampling rates? Or rather go to a pretty high frequency to preserve variance in the signal? Going with the latter (10min) for now.

# %% hidden=true
time_span = niskine.io.mooring_start_end_time(mooring=1)
tnew = np.arange(time_span.start, time_span.stop, 10, dtype='datetime64[m]')
# Run the following line for second precision of the time vector (should not matter...)
tnew = np.array([np.datetime64(ti, 's') for ti in tnew])

# %% hidden=true
tnew[:3]

# %% hidden=true
asort_ti = niskine.merge.interpolate_time(asort, tnew)

# %% [markdown] hidden=true
# #### Try other resampling method

# %% [markdown] hidden=true
# What about xarrays `resample`? Try this and compare to interpolation. When using `interpolate`, the result is identical. It seems like a good idea to provide our own new time vector so it is the same for all, so let's stick with the interpolation above.  

# %% hidden=true
# Pick one level to test
test = asort[1].u.sel(z=500, time=slice('2019-06-01', '2019-06-01'))
testi = test.interp(time=tnew)
tnew_slow = np.arange(time_span.start, time_span.stop, 32, dtype='datetime64[m]')
testii = test.interp(time=tnew_slow)
testr = test.resample(time="1H", loffset=np.timedelta64(0, 'm')).mean()
testri = test.resample(time="5min").interpolate('linear')

# %% hidden=true
test.plot()
testi.plot()
testii.plot()
# testri.plot()
gv.plot.png('time_interpolation.png')

# %% hidden=true
for ti in [test, testi, testii, testr, testri]:
    print(ti.var().data)

# %% [markdown] heading_collapsed=true
# ### Interpolate depth

# %% [markdown] hidden=true
# Ideally, we would have set the depth vector to already be pretty much the same for all ADCPs in the processing. I can still go back and do this. For now, just interpolate to a 16m depth vector that starts at 0.

# %% hidden=true
znew = np.arange(0, 2016, 16)

# %% hidden=true
asort_ti_zi = niskine.merge.interpolate_depth(asort_ti, znew)

# %% hidden=true
fig, ax = gv.plot.quickfig()
opts = dict(vmin=-0.5, vmax=0.5, add_colorbar=False, cmap='RdBu_r')
for ci in asort_ti_zi:
    ci.u.gv.tcoarsen().plot(ax=ax, **opts);
    ax.invert_yaxis()

# %% hidden=true
fig, ax = gv.plot.quickfig()
opts = dict(vmin=-0.5, vmax=0.5, add_colorbar=False, cmap='RdBu_r')
for ci in asort_ti_zi:
    ci.u.sel(time='2020-01').plot(ax=ax, **opts);
    ax.invert_yaxis()

# %% [markdown] heading_collapsed=true
# ### Merge into one dataset

# %% [markdown] hidden=true
# #### Determine Overlap

# %% [markdown] hidden=true
# We can probably merge M1 without interpolation, but let's find out how much overlap there is.

# %% hidden=true
overlap = niskine.merge.determine_overlap(asort_ti_zi)

# %% hidden=true
overlap.sum(dim='adcp').plot()

# %% hidden=true
overlap.sum(dim='adcp').max(dim='z').plot()

# %% hidden=true
h, bins, hh = overlap.sum(dim='adcp').max(dim='z').plot.hist(bins=[0, 1, 2, 3])

# %% hidden=true
print(f'overlaps {int(h[2])} times')

# %% [markdown] hidden=true
# #### Merge

# %% hidden=true
d_simple = niskine.merge.simple_merge(asort_ti_zi)
d_simple = niskine.merge.add_mooring_metadata(d_simple, mooring=1)

# %% hidden=true
d_median = niskine.merge.median_merge(asort_ti_zi)
d_median = niskine.merge.add_mooring_metadata(d_median, mooring=1)

# %% [markdown] hidden=true
# Save without filled gaps (now saving everything in the other notebook).

# %% hidden=true
conf = niskine.io.load_config()
conf.data.gridded.adcp.mkdir(exist_ok=True, parents=True)
# niskine.merge.save_merged(d_median, suffix='median_merge')

# %% [markdown] hidden=true
# Fill gaps

# %% hidden=true
df_median = niskine.merge.fill_gaps(d_median)

# %% hidden=true
# niskine.merge.save_merged(df_median, suffix='median_merge_gaps_filled')

# %% hidden=true
df_median.u.sel(time='2020-02').gv.tplot()

# %% [markdown] hidden=true
# Add ADCP depth, temperature

# %% hidden=true
df_median = niskine.merge.add_auxilliary_data(asort_ti_zi, df_median)

# %% hidden=true
df_median.temperature.plot(hue='adcp', color='k', linewidth=0.5)

# %% hidden=true
ax = df_median.u.gv.tcoarsen().gv.tplot()
df_median.xducer_depth.gv.tcoarsen().plot(ax=ax, hue='adcp', color='k', linewidth=0.5)
