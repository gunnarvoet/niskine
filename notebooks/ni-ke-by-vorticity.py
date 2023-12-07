# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
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
mpl.rcParams["lines.linewidth"] = 1

# %% [markdown]
# # NISKINe NI KE sorted by surface vorticity

# %%
rho0 = 1025

# %%
# load mooring locations
locs = xr.open_dataset(cfg.mooring_locations)
locs.close()

# %%
# load gridded temperature (update: do not think we need this here)
# tall = xr.open_dataarray(niskine.io.CFG.data.gridded.temperature)
# subsample to daily averages
# td = tall.resample(time="1D").mean()

# %%
# load vorticity & EKE at M1
alt = xr.open_dataset(cfg.data.ssh_m1)
alt.close()

# %%
# load NI EKE
ni_eke = xr.open_dataarray(cfg.data.ni_eke_m1)
ni_eke.close()

# %%
# interpolate vorticity to NI EKE
vorti = alt.vort.interp_like(ni_eke)

# %%
# define masks for positive and negative vorticity
mask_pos = vorti > 0
mask_neg = vorti < 0

# %%
# load wind stress (N/m^2)
wind_stress = xr.open_dataarray(cfg.data.wind_work.niskine_m1_wind_stress).interp_like(
    ni_eke
)
wind_stress.close()
# There are a few values at the edges that end up being nan's, let's just set them to zero so we don't have to deal with them. It's really only less than an hour on each side from the interpolation above.
wind_stress[np.isnan(wind_stress)] = 0

# %%
percentile = 50
print(f"wind stress {percentile} percentile: {np.percentile(wind_stress, 50):1.2f} N/m^2")

# %%
mask_high_wind_stress = wind_stress > 0.1
mask_low_wind_stress = wind_stress < 0.1

# %% [markdown]
# Drop depth levels where we have only very little NI EKE data.

# %%
maskna = (~np.isnan(ni_eke)).sum(dim="time") > 7e4

fig, ax = gv.plot.quickfig(w=3)
h = (~np.isnan(ni_eke)).sum(dim="time").plot(yincrease=False, y="z")
h = (
    (~np.isnan(ni_eke))
    .sum(dim="time")
    .where(maskna)
    .plot(yincrease=False, y="z", linewidth=3, color="C6")
)

# %%
# divide into positive and negative vorticity
mp = ni_eke.where(mask_pos & maskna).mean(dim="time")
mn = ni_eke.where(mask_neg & maskna).mean(dim="time")

# %%
# divide into positive and negative vorticity and only high wind stress
mp_hws = ni_eke.where(mask_pos & maskna & mask_high_wind_stress).mean(dim="time")
mn_hws = ni_eke.where(mask_neg & maskna & mask_high_wind_stress).mean(dim="time")

# divide into positive and negative vorticity and low wind stress
mp_lws = ni_eke.where(mask_pos & maskna & mask_low_wind_stress).mean(dim="time")
mn_lws = ni_eke.where(mask_neg & maskna & mask_low_wind_stress).mean(dim="time")

# %%
fig, ax = gv.plot.quickfig(h=5, w=4, grid=True)
mp.plot(color="C0", ax=ax, y="z", yincrease=False, label=r"$\zeta/f > 0$")
mp_hws.plot(color="C0", ax=ax, y="z", yincrease=False, linestyle="--")
mp_lws.plot(color="C0", ax=ax, y="z", yincrease=False, linestyle="-.")
mn.plot(color="C6", ax=ax, y="z", yincrease=False, label=r"$\zeta/f < 0$")
mn_hws.plot(color="C6", ax=ax, y="z", yincrease=False, linestyle="--")
mn_lws.plot(color="C6", ax=ax, y="z", yincrease=False, linestyle="-.")
ax.legend()
ax.set(title="M1 $\mathrm{KE}_\mathrm{NI}$", xlabel="$\mathrm{KE}_\mathrm{NI}$ [J/m$^3$]");

# %% [markdown]
# How do we determine the uncertainty of this estimate? Obviously, the standard deviation contains lots of natural variability. Maybe we can use a bootstrap method and calculate the mean from randomly selected subsets?

# %% [markdown]
# Actually, the standard error is defined as $\pm\frac{\sigma}{\sqrt{n}}$ where $\sigma$ is the standard deviation and $n$ is the degrees of freedom in the dataset. For 95% confidence intervals this becomes $\pm1.96\, \frac{\sigma}{\sqrt{n}}$.

# %%
# number of days in the time series
ndays = (ni_eke.time[-1] - ni_eke.time[0]).data.astype('timedelta64[D]')
print(f"length of time series: {ndays}")

# %%
ndays

# %% [markdown]
# What is the decorrelation time scale?

# %%
ni_eke_daily = ni_eke.resample(time="d").mean()
vorti_daily = vorti.resample(time="d").mean()

fig, ax = gv.plot.quickfig()
ax.acorr(ni_eke_daily.sel(z=400, method="nearest"), usevlines=True, normed=True, maxlags=50, lw=1)
ax.vlines([-7, 7], ymin=0, ymax=1, color="C6", linewidth=2)
ax.grid(True)

# %% [markdown]
# From the plot above a decorrelation time scale of 7 days seems to be a good estimate.

# %%
decorrelation_time_scale = np.timedelta64(7, "D")

# %%
dof = ndays / decorrelation_time_scale
print(f"dof: {dof:1.0f}")

# %%
mp_std = ni_eke.where(mask_pos & maskna).std(dim="time")
mp_se = mp_std / np.sqrt(dof)
mn_std = ni_eke.where(mask_neg & maskna).std(dim="time")
mn_se = mn_std / np.sqrt(dof)

# %% [markdown]
# What is the decorrelation time scale? See below the autocorrelation plot of daily averaged NI EKE, it suggests a decorrelation time scale of about 10 days. At a length of 500 days we thus have $n=50$ degrees of freedom.

# %% [markdown]
# Plot with 1 standard error

# %%
fig, ax = gv.plot.quickfig(h=5, w=4, grid=True)

ax.fill_betweenx(mp.z, mp-1*mp_se, mp+1*mp_se, color="C3", alpha=0.2, edgecolor=None)
ax.fill_betweenx(mn.z, mn-1*mn_se, mn+1*mn_se, color="C0", alpha=0.2, edgecolor=None)

mp.plot(color="w", linewidth=2, ax=ax, y="z", yincrease=False)
mp.plot(color="C3", ax=ax, y="z", yincrease=False, label=r"$\zeta/f > 0$")
mp_hws.plot(color="C3", ax=ax, y="z", yincrease=False, linestyle="--")
mp_lws.plot(color="C3", ax=ax, y="z", yincrease=False, linestyle="-.")

mn.plot(color="w", linewidth=2, ax=ax, y="z", yincrease=False)
mn.plot(color="C0", ax=ax, y="z", yincrease=False, label=r"$\zeta/f < 0$")
mn_hws.plot(color="C0", ax=ax, y="z", yincrease=False, linestyle="--")
mn_lws.plot(color="C0", ax=ax, y="z", yincrease=False, linestyle="-.")

ax.legend(loc="center right")
ax.set(title="M1 $\mathrm{KE}_\mathrm{NI}$", xlabel="$\mathrm{KE}_\mathrm{NI}$ [J/m$^3$]");
niskine.io.png("m1_NI_KE_by_vorticity_with_1_standard_error")

# %% [markdown]
# Plot with 2 standard errors

# %%
fig, ax = gv.plot.quickfig(h=5, w=4, grid=True)

ax.fill_betweenx(mp.z, mp-1.96*mp_se, mp+1.96*mp_se, color="C3", alpha=0.2, edgecolor=None)
ax.fill_betweenx(mn.z, mn-1.95*mn_se, mn+1.95*mn_se, color="C0", alpha=0.2, edgecolor=None)

mp.plot(color="w", linewidth=2, ax=ax, y="z", yincrease=False)
mp.plot(color="C3", ax=ax, y="z", yincrease=False, label=r"$\zeta/f > 0$")
mp_hws.plot(color="w", ax=ax, y="z", yincrease=False, linewidth=2)
mp_hws.plot(color="C3", ax=ax, y="z", yincrease=False, linestyle="--")
mp_lws.plot(color="w", ax=ax, y="z", yincrease=False, linewidth=2)
mp_lws.plot(color="C3", ax=ax, y="z", yincrease=False, linestyle="-.")


mn.plot(color="w", linewidth=2, ax=ax, y="z", yincrease=False)
mn.plot(color="C0", ax=ax, y="z", yincrease=False, label=r"$\zeta/f < 0$")
mn_hws.plot(color="w", ax=ax, y="z", yincrease=False, linewidth=2)
mn_hws.plot(color="C0", ax=ax, y="z", yincrease=False, linestyle="--")
mn_lws.plot(color="w", ax=ax, y="z", yincrease=False, linewidth=2)
mn_lws.plot(color="C0", ax=ax, y="z", yincrease=False, linestyle="-.")

ax.legend(loc="center right")
ax.set(title="M1 $\mathrm{KE}_\mathrm{NI}$", xlabel="$\mathrm{KE}_\mathrm{NI}$ [J/m$^3$]");
niskine.io.png("m1_NI_KE_by_vorticity_with_2_standard_error")
niskine.io.pdf("m1_NI_KE_by_vorticity_with_2_standard_error")


# %% [markdown]
# ---

# %% [markdown]
# ---

# %% [markdown]
# bootstrapping (not using this anymore)

# %% [markdown]
# Note: We possibly want to calculate daily averages first before bootstrapping, otherwise we still have so many of the same events. Or maybe even average over several days.

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
# niskine.io.png("m1_NI_KE_by_vorticity_with_2sig")
