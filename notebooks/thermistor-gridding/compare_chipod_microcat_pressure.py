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
# ## Compare chipod & microcat pressure

# %% [markdown]
# ### Time Vector

# %% [markdown]
# Create a universal time vector for M1. We will go with hourly and interpolate all chipod and microcat pressure time series to this universal time vector.
#
# Let's only run through May 2020 as chipods and microcats start dropping out. Actually, some Microcats start giving up earlier in the year but maybe they are not as important when we have the chipod pressure time series. We may even be able to figure out the general behavior of the mooring during knockdown, come up with a mapping, and get away with only a few pressure measurements at the end of the time series.
#
# Update: If we want stratification until the end of the deployment we have to work with microcats only as the chipods all drop out around May. We now fit a second order polynomial to the knockdown characteristics in the microcat data. Note: This could also be done with the chipod pressure time series for finer resolution but there seem to be some issues with the chipod pressure that I don't want to deal with at the moment.

# %%
timeslice = niskine.io.mooring_start_end_time(mooring=1)
print(timeslice.start, "  --  ", timeslice.stop)

# %%
time = np.arange(timeslice.start, np.datetime64("2020-05-01 12:00:00"), dtype="datetime64[20m]").astype(
    "datetime64[m]"
)

# %%
mm = niskine.io.read_m1_sensor_config()
# drop the chipod that has bad pressure
mm = mm.drop(index=614)

# %% [markdown]
# Extract info for all instruments with pressure sensor.

# %%
# serial numbers
psn = mm.where((mm.sensor=="chi") | (mm.sensor=="ctd")).dropna(how='all').index.to_numpy()
# full info
pmm = mm.where((mm.sensor=="chi") | (mm.sensor=="ctd")).dropna(how='all')

# %% [markdown]
# ### Chipod pressure time series

# %% [markdown]
# Chipods to remove (there may be more issues...):
# - 630 (no pressure)
# - 614 (funky pressure)

# %%
cc = niskine.strat.read_chipod_pressure(time)

# %%
cc.gv.plot(hue="sn", alpha=0.5, linewidth=0.5, yincrease=False);

# %% [markdown]
# ### Microcat pressure time series

# %%
s = niskine.io.read_microcats(common_time=time)
# extract pressure only
sp = s.p
sp["mean_pressure"] = sp.mean(dim='time')
sp = sp.sortby("mean_pressure")

# %% [markdown]
# ### Compare chipod and microcat pressure

# %%
fig, ax = gv.plot.quickfig(h=3)
time_slice = slice("2019-11-27", "2019-12-02")
ds = sp.sel(time=time_slice)
ds.isel(sn=range(4)).gv.plot(
    hue="sn", linewidth=0.7, yincrease=False, ax=ax, color="0.2"
)
for g, dsi in ds.isel(sn=range(4)).groupby("mp"):
    ax.text(
        dsi.time.isel(time=-1).data + np.timedelta64(4, "h"),
        dsi.isel(time=-1).data,
        f"{dsi.sn.item()}",
        color=f"0.2",
        va="center",
        fontsize=9,
        fontweight="normal",
    )
# also plot one or two chipod pressure time series
# for csni in [624, 627, 632]:
#     cc.sel(sn=csni, time=time_slice).gv.plot(ax=ax, color="C0", linewidth=0.5)
for g, ci in cc.groupby("sn"):
    ci.sel(time=time_slice).gv.plot(ax=ax, color="C0", linewidth=0.5)

# %%
fig, ax = gv.plot.quickfig(h=3)
time_slice = slice("2020-03-02", "2020-03-03")
ds = sp.sel(time=time_slice)
ds.isel(sn=range(4)).gv.plot(
    hue="sn", linewidth=0.7, yincrease=False, ax=ax, color="0.2"
)
# for g, dsi in ds.isel(sn=range(4)).groupby("mp"):
#     ax.text(
#         dsi.time.isel(time=-1).data + np.timedelta64(4, "h"),
#         dsi.isel(time=-1).data,
#         f"{dsi.sn.item()}",
#         color=f"0.2",
#         va="center",
#         fontsize=9,
#         fontweight="normal",
#     )
# also plot one or two chipod pressure time series
# for csni in [624, 627, 632]:
#     cc.sel(sn=csni, time=time_slice).gv.plot(ax=ax, color="C0", linewidth=0.5)
for g, ci in cc.groupby("sn"):
    ci.sel(time=time_slice).gv.plot(ax=ax, color="C0", linewidth=0.5)

# %%
fig, ax = gv.plot.quickfig(grid=True)
sp.gv.plot(hue="sn", yincrease=False, ax=ax)
cc.gv.plot(hue="sn", yincrease=False, ax=ax)

# %% [markdown]
# ## Join chipod and microcat pressure datasets

# %%
p = xr.concat([cc, sp], dim="sn")

# %% [markdown]
# How to plot one time step?

# %%
data = p.isel(time=1000)
fig, ax = gv.plot.quickfig(w=2)
ax.plot(np.ones_like(data), data, marker='x', linestyle="")
ax.invert_yaxis()
ax.set(yscale="log")

# %% [markdown]
# Chipod 614 seems a bit shallow at the end - compare to microcat below. We should throw this out or at least cut the time series...

# %%
time = "2019-08"
fig, ax = gv.plot.quickfig()
p.sel(sn=614, time=time).plot()
p.sel(sn=4923, time=time).plot()
ax.invert_yaxis()

# %%
time = "2019-10-07"
fig, ax = gv.plot.quickfig()
p.sel(sn=616, time=time).plot()
p.sel(sn=4923, time=time).plot()
ax.invert_yaxis()


# %%
def compare_pressure_differences(base_sn):
    fig, ax = gv.plot.quickfig()
    for sni in p.sn:
        if p.sel(sn=sni).mean()<1000:
            diffp = p.sel(sn=sni)-p.sel(sn=base_sn)
            (diffp - diffp.mean()).plot()
    ax.invert_yaxis()
#     ax.set(ylim=[-100, 500])


# %%
compare_pressure_differences(base_sn=618)

# %%
fig, ax = gv.plot.quickfig()
(p.sel(sn=614)-p.sel(sn=618)).plot()
(p.sel(sn=615)-p.sel(sn=618)).plot()
(p.sel(sn=616)-p.sel(sn=618)).plot()
(p.sel(sn=617)-p.sel(sn=618)).plot()
(p.sel(sn=619)-p.sel(sn=618)).plot()
(p.sel(sn=620)-p.sel(sn=618)).plot()
(p.sel(sn=621)-p.sel(sn=618)).plot()
(p.sel(sn=622)-p.sel(sn=618)).plot()
(p.sel(sn=623)-p.sel(sn=618)).plot()
(p.sel(sn=624)-p.sel(sn=618)).plot()
(p.sel(sn=625)-p.sel(sn=618)).plot()
ax.invert_yaxis()

# %%
fig, ax = gv.plot.quickfig()
(p.sel(sn=614)-p.sel(sn=619)).plot()
(p.sel(sn=615)-p.sel(sn=619)).plot()
(p.sel(sn=616)-p.sel(sn=619)).plot()
(p.sel(sn=617)-p.sel(sn=619)).plot()
(p.sel(sn=618)-p.sel(sn=619)).plot()
(p.sel(sn=620)-p.sel(sn=619)).plot()
(p.sel(sn=621)-p.sel(sn=619)).plot()
(p.sel(sn=622)-p.sel(sn=619)).plot()
(p.sel(sn=623)-p.sel(sn=619)).plot()
(p.sel(sn=624)-p.sel(sn=619)).plot()
(p.sel(sn=625)-p.sel(sn=619)).plot()
ax.invert_yaxis()

# %%
fig, ax = gv.plot.quickfig()
(p.sel(sn=614)-p.sel(sn=620)).plot()
(p.sel(sn=615)-p.sel(sn=620)).plot()
(p.sel(sn=616)-p.sel(sn=620)).plot()
(p.sel(sn=617)-p.sel(sn=620)).plot()
(p.sel(sn=618)-p.sel(sn=620)).plot()
(p.sel(sn=619)-p.sel(sn=620)).plot()
ax.invert_yaxis()

# %%
fig, ax = gv.plot.quickfig()
(p.sel(sn=614)-p.sel(sn=621)).plot()
(p.sel(sn=615)-p.sel(sn=621)).plot()
(p.sel(sn=4923)-p.sel(sn=621)).plot()
(p.sel(sn=616)-p.sel(sn=621)).plot()
(p.sel(sn=617)-p.sel(sn=621)).plot()
(p.sel(sn=618)-p.sel(sn=621)).plot()
(p.sel(sn=619)-p.sel(sn=621)).plot()
(p.sel(sn=620)-p.sel(sn=621)).plot()
(p.sel(sn=620)-p.sel(sn=621)).plot()
(p.sel(sn=622)-p.sel(sn=621)).plot()
(p.sel(sn=623)-p.sel(sn=621)).plot()
(p.sel(sn=2864)-p.sel(sn=621)).plot()
ax.invert_yaxis()

# %% [markdown]
# ### Compare measured pressure to nominal depth

# %%
sn = 4923
dep = mm.loc[sn].depth
data = p.sel(sn=sn)


# %%
def compare_nominal_and_measured_depth(sn, verbose=False):
    data = p.sel(sn=sn, time="2019")
    nominal = mm.loc[sn].depth
    minimum = np.absolute(gsw.z_from_p(data.min().item(), lat))
    half_percentile = np.absolute(gsw.z_from_p(np.percentile(data, 0.2), lat))
    if verbose:
        print("nominal:", nominal)
        print("minimum:", minimum)
        print("0.5 percentile:", half_percentile)
    return nominal, nominal - half_percentile


# %% [markdown]
# Comparing nominal depth with measured depth for all SBE37 (except the really deep one since it is not on the spreadsheet) shows that they are pretty spot on (within a meter).

# %%
sbe37sn = mm.where(mm.sensor=='ctd').dropna().index.to_numpy()
diffs = []
for sn in sbe37sn:
    diffs.append(compare_nominal_and_measured_depth(sn))

# %%
diffs

# %%
fig, ax = gv.plot.quickfig()
for sni in sbe37sn:
    p.sel(sn=sni).plot.hist(bins='auto', density=True, alpha=0.3, ax=ax);
ax.set(title='');

# %%
chisn = mm.where(mm.sensor=="chi").dropna(how='all').index.to_numpy()
chidiffs = []
for sn in chisn:
    chidiffs.append(compare_nominal_and_measured_depth(sn))

# %%
chidiffs

# %%
fig, ax = gv.plot.quickfig()
for sni in chisn:
    p.sel(sn=sni).plot.hist(bins='auto', density=True, alpha=0.3, ax=ax);
ax.set(title='');

# %% [markdown]
# Trying to fit a skewed distribution to the data but the cutoff at the resting position is too sharp:

# %%
data = p.sel(sn=4923)

from scipy import stats
ae, loce, scalee = stats.skewnorm.fit(p.sel(sn=sn).data)
h = p.sel(sn=sn).plot.hist(density=True, bins='auto', alpha=0.5)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
skewp = stats.skewnorm.pdf(x,ae, loce, scalee)#.rvs(100)
plt.plot(x, skewp, 'k', linewidth=2)
