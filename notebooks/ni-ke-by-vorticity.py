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
# # %load /Users/gunnar/Projects/python/standard_imports.py
# %matplotlib inline
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from pathlib import Path
import random

import gvpy as gv

import niskine

# %reload_ext autoreload
# %autoreload 2

# %config InlineBackend.figure_format = 'retina'

# %%
cfg = niskine.io.load_config()

# %%
gv.plot.helvetica()

# %% [markdown]
# # NISKINe Plot Mean NI KE sorted by surface vorticity

# %%
rho0 = 1025

# %%
# load mooring locations
locs = xr.open_dataset(cfg.mooring_locations)
locs.close()

# %%
# load gridded temperature
tall = xr.open_dataarray(niskine.io.CFG.data.gridded.temperature)
# subsample to daily averages
td = tall.resample(time="1D").mean()

# %%
# load vorticity & EKE at M1
alt = xr.open_dataset(cfg.data.ssh_m1)
alt.close()

# %%
# load NI EKE
ni_eke = xr.open_dataarray(cfg.data.ni_eke_m1)
ni_eke.close()

# %%
vorti = alt.vort.interp_like(ni_eke)

# %%
mask_pos = vorti > 0
mask_neg = vorti < 0

# %% [markdown]
# drop depth levels with only very little data!

# %%
(~np.isnan(ni_eke)).sum(dim="time").plot(yincrease=False, y="z")
maskna = (~np.isnan(ni_eke)).sum(dim="time") > 7e4
(~np.isnan(ni_eke)).sum(dim="time").where(maskna).plot(yincrease=False, y="z")

# %%
fig, ax = gv.plot.quickfig(h=5, w=4, grid=True)
ni_eke.where(mask_pos & maskna).mean(dim="time").plot(ax=ax, y="z", yincrease=False, label=r"$\zeta/f > 0$")
ni_eke.where(mask_neg & maskna).mean(dim="time").plot(ax=ax, y="z", yincrease=False, label=r"$\zeta/f < 0$")
ax.legend()
ax.set(title="M1 $\mathrm{KE}_\mathrm{NI}$", xlabel="$\mathrm{KE}_\mathrm{NI}$ [J/m$^3$]");
niskine.io.png("m1_NI_KE_by_vorticity")

# %% [markdown]
# How do we determine the uncertainty of this estimate? Obviously, the standard deviation contains lots of natural variability. Maybe we can use a bootstrap method and calculate the mean from randomly selected subsets?

# %%
fig, ax = gv.plot.quickfig(h=5, w=4)
ni_eke.where(mask_pos & maskna).std(dim="time").plot(ax=ax, y="z", yincrease=False, label=r"$\zeta/f > 0$")
ni_eke.where(mask_neg & maskna).std(dim="time").plot(ax=ax, y="z", yincrease=False, label=r"$\zeta/f < 0$")
ax.legend()
ax.set(title="M1 std($\mathrm{KE}_\mathrm{NI}$)");

# %% [markdown]
# Note: We possibly want to calculate daily averages first before bootstrapping, otherwise we still have so many of the same events. Or maybe even average over several days.

# %%
ni_eke_weekly = ni_eke.resample(time="7d").mean()
ni_eke_daily = ni_eke.resample(time="d").mean()
vorti_daily = vorti.resample(time="d").mean()


# %%
def mean_from_random_selection(seed):
    n = len(ni_eke_daily.time)
    nr = list(range(n))
    sel_ind = np.array(sorted(random.choices(nr, k=n)))
    ni_ke_sel = ni_eke_daily.isel(time=sel_ind)
    vort_sel = vorti_daily.isel(time=sel_ind)
    mask_pos = vort_sel > 0
    mask_neg = vort_sel < 0
    mp = ni_ke_sel.where(mask_pos & maskna).mean(dim="time")
    mn = ni_ke_sel.where(mask_neg & maskna).mean(dim="time")
    return mp, mn 


# %%
mp, mn = mean_from_random_selection(1)

# %%
out = [mean_from_random_selection(i) for i in range(1000)]

# %%
allp = [outi[0] for outi in out]
alln = [outi[1] for outi in out]

# %%
mean_pos_vort = xr.concat(allp, dim="n")
mp_mean = mean_pos_vort.mean(dim="n")
mp_std = mean_pos_vort.std(dim="n")
mean_neg_vort = xr.concat(alln, dim="n")
mn_mean = mean_neg_vort.mean(dim="n")
mn_std = mean_neg_vort.std(dim="n")

# %%
fig, ax = gv.plot.quickfig(h=5, w=4, grid=True)

ax.fill_betweenx(mp_std.z, mp_mean-2*mp_std, mp_mean+2*mp_std, color="C3", alpha=0.2)
mp_mean.plot(color="C3", ax=ax, y="z", yincrease=False, label=r"$\zeta/f > 0$")

ax.fill_betweenx(mn_std.z, mn_mean-2*mn_std, mn_mean+2*mn_std, color="C0", alpha=0.2)
mn_mean.plot(color="C0", ax=ax, y="z", yincrease=False, label=r"$\zeta/f < 0$")

ax.legend(loc="center right")
ax.set(title="M1 $\mathrm{KE}_\mathrm{NI}$", xlabel="$\mathrm{KE}_\mathrm{NI}$ [J/m$^3$]");
niskine.io.png("m1_NI_KE_by_vorticity_with_2sig")
