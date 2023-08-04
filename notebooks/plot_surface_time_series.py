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
# # NISKINe Plot Surface Time Series

# %% [markdown]
# Make a plot with panels for
# - vorticity & EKE
# - wind stress & NI wind work

# %%
rho0 = 1025

# %%
# load mooring locations
locs = xr.open_dataset(cfg.mooring_locations)
locs.close()

# %%
# load gridded temperature
tall = xr.open_dataarray(niskine.io.CFG.data.gridded.temperature)

# %%
xr.__version__

# %%
# subsample to daily averages
td = tall.resample(time="1D").mean()

# %%
td.gv.tplot()

# %%
# load ADCP data
adcp = niskine.io.load_gridded_adcp(mooring=1)
adcp.close()

# %%
ud = adcp.u.resample(time="1D").mean()

# %%
m1eke = 0.5*rho0*(adcp.u**2 + adcp.v**2)
m1eke.gv.tcoarsen().gv.tplot(robust=True)

# %%
# load MLD vel
mldvel = xr.open_dataset(cfg.data.ml.ml_vel)
mldvel.close()
# load MLD
mld = xr.open_dataarray(cfg.data.ml.mld)
mld.close()

# %%
# calculate mixed layer KE and EKE
tlong, tshort = niskine.calcs.determine_ni_band(1.06, )
lp_period_hours = 72
ulp = niskine.calcs.lowpass_time_series(mldvel.u, lp_period_hours, fs=6)
vlp = niskine.calcs.lowpass_time_series(mldvel.v, lp_period_hours, fs=6)
mld_eke = 0.5 * (ulp**2 + vlp **2)
mldvel["eke"] = (("time"), mld_eke)
mldvel["ke"] = (("time"), (0.5 * (mldvel.u**2 + mldvel.v**2)).data)

# %%
# load vorticity & EKE at M1
alt = xr.open_dataset(cfg.data.ssh_m1)
alt.close()

# %%
# load wind work
wind_work = xr.open_dataarray(cfg.data.wind_work.niskine_m1)
wind_work.close()
wind_work_c = xr.open_dataarray(cfg.data.wind_work.niskine_m1_cumulative)
wind_work_c.close()
wind_stress = xr.open_dataarray(cfg.data.wind_work.niskine_m1_wind_stress)
wind_stress.close()

# %%
fig, ax = gv.plot.quickfig(w=7, h=2.5)
mldvel.ke.plot(ax=ax, color="0.7", linewidth=1)
mldvel.eke.plot(ax=ax, color="w", linewidth=1.5)
mldvel.eke.plot(ax=ax, color="C4", linewidth=1)
alt.eke.sel(time=slice(wind_work.time[0].data, wind_work.time[-1].data)).plot(ax=ax, color="w", linewidth=1.5)
alt.eke.sel(time=slice(wind_work.time[0].data, wind_work.time[-1].data)).plot(ax=ax, color="C0", linewidth=1)
# alt.eke.groupby('time.month').mean('time').plot(linestyle="", marker="o")
ax.set(xlabel="", ylabel="surface EKE [m$^2$ s$^{-2}$]", title="")
gv.plot.concise_date(ax)

# %%
# load NI EKE
ni_eke = xr.open_dataarray(cfg.data.ni_eke_m1)
ni_eke.close()

# %%
ni_eke.mean(dim="time").plot(y="z", yincrease=False)

# %%
fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(7.5, 7),
                       constrained_layout=True, sharex=True,
                       height_ratios=[1, 1, 1, 1, 1, 2.5])

wind_stress.plot(ax=ax[0], linewidth=1, color="0.1")
ax[0].set(ylabel="[N m$^{-2}$]")
ax[0].annotate(r"wind stress $\tau$", xy=(0.95, 0.8), xycoords="axes fraction", backgroundcolor="w", ha="right")

(wind_work*1e3).plot(ax=ax[1], linewidth=1, color="0.1")
ax[1].set(ylabel="[mW m$^{-2}$]")
ax[1].annotate(r"NI wind work $\Pi_\mathrm{NI}$", xy=(0.95, 0.8), xycoords="axes fraction", backgroundcolor="w", ha="right")


wind_work_c.plot(ax=ax[2], linewidth=1, color="0.1")
ax[2].set(ylabel="[kJ m$^{-2}$]")
ax[2].annotate(r"cumulative NI wind work $\quad\int \Pi_\mathrm{NI} dt$", xy=(0.95, 0.1), xycoords="axes fraction", backgroundcolor="w", ha="right")

alt.vort.sel(time=slice(wind_work.time[0].data, wind_work.time[-1].data)).plot(ax=ax[3], linewidth=1, color="0.1")
ax[3].set(xlabel="", ylabel="")
ax[3].annotate(r"vorticity $\zeta/f$", xy=(0.95, 0.1), xycoords="axes fraction", backgroundcolor="w", ha="right")

mldvel.ke.plot(ax=ax[4], color="0.7", linewidth=1)
mldvel.eke.plot(ax=ax[4], color="w", linewidth=1.5)
mldvel.eke.plot(ax=ax[4], color="C4", linewidth=1)
alt.eke.sel(time=slice(wind_work.time[0].data, wind_work.time[-1].data)).plot(ax=ax[4], color="w", linewidth=1.5)
alt.eke.sel(time=slice(wind_work.time[0].data, wind_work.time[-1].data)).plot(ax=ax[4], linewidth=1)
ax[4].set(xlabel="", ylabel="[m$^2$ s$^{-2}$]")
ax[4].annotate("KE & EKE", xy=(0.95, 0.75), xycoords="axes fraction", backgroundcolor="w", ha="right")
ax[4].annotate("ML KE", xy=(0.17, 0.85), xycoords="axes fraction", ha="left", color="0.5", fontsize=7)
ax[4].annotate("ML EKE", xy=(0.18, 0.55), xycoords="axes fraction", ha="left", color="C4", fontsize=7)
ax[4].annotate("SSH EKE", xy=(0.65, -0.05), xycoords="axes fraction", ha="left", color="C0", fontsize=7)

h = ni_eke.plot.contourf(ax=ax[5], extend="both", levels=np.arange(0.25, 4, 0.25), cmap="Spectral_r", cbar_kwargs=dict(aspect=30, shrink=0.7, pad=0.02, label="[J m$^{-3}$]", ticks=mpl.ticker.MaxNLocator(4)))
gv.plot.contourf_hide_edges(h)
ax[5] = mld.gv.tcoarsen(n=30*24*2).gv.tplot(ax=ax[5], color="w", linewidth=1.75, alpha=1)
ax[5] = mld.gv.tcoarsen(n=30*24*2).gv.tplot(ax=ax[5], color="k", linewidth=1, alpha=1)
ax[5].invert_yaxis()
ax[5].set(xlabel="", ylabel="depth [m]")
ax[5].annotate("NI KE", xy=(0.95, 0.1), xycoords="axes fraction", backgroundcolor="w", ha="right")

ax[5].set(xlim=[np.datetime64("2019-05-01"), np.datetime64("2020-10-10")])

gv.plot.subplotlabel(ax[:-1], x=0.01, y=0.85, fs=10)
# gv.plot.subplotlabel(ax[-1:], x=0.01, y=0.9, )
ax[-1].annotate("f", (0.01, 0.9), xycoords="axes fraction", fontsize=10)

for axi in ax:
    gv.plot.axstyle(axi)
    axi.set(title="")
gv.plot.concise_date_all()

plot_fig = True
if plot_fig:
    name = "surface_forcing"
    niskine.io.png(name)
    niskine.io.pdf(name)

# %%
fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(8, 8),
                       constrained_layout=True, sharex=True,
                       height_ratios=[1, 1, 1, 1, 1, 2.5, 2.5])

wind_stress.plot(ax=ax[0])
ax[0].set(ylabel="[N m$^{-2}$]")
ax[0].annotate(r"wind stress $\tau$", xy=(0.95, 0.8), xycoords="axes fraction", backgroundcolor="w", ha="right")

(wind_work*1e3).plot(ax=ax[1])
ax[1].set(ylabel="[mW m$^{-2}$]")
ax[1].annotate(r"NI wind work $\Pi_\mathrm{NI}$", xy=(0.95, 0.8), xycoords="axes fraction", backgroundcolor="w", ha="right")


wind_work_c.plot(ax=ax[2])
ax[2].set(ylabel="[kJ m$^{-2}$]")
ax[2].annotate(r"cumulative NI wind work $\quad\int \Pi_\mathrm{NI} dt$", xy=(0.95, 0.1), xycoords="axes fraction", backgroundcolor="w", ha="right")

alt.vort.sel(time=slice(wind_work.time[0].data, wind_work.time[-1].data)).plot(ax=ax[3])
ax[3].set(xlabel="", ylabel="")
ax[3].annotate(r"vorticity $\zeta/f$", xy=(0.95, 0.1), xycoords="axes fraction", backgroundcolor="w", ha="right")

mldvel.ke.plot(ax=ax[4], color="0.8", linewidth=1)
mldvel.eke.plot(ax=ax[4], color="C4", linewidth=1)
alt.eke.sel(time=slice(wind_work.time[0].data, wind_work.time[-1].data)).plot(ax=ax[4])
ax[4].set(xlabel="", ylabel="[m$^2$ s$^{-2}$]")
ax[4].annotate("EKE", xy=(0.95, 0.75), xycoords="axes fraction", backgroundcolor="w", ha="right")

h = ni_eke.plot.contourf(ax=ax[5], extend="both", levels=np.arange(0.25, 4, 0.25), cmap="Spectral_r", cbar_kwargs=dict(aspect=30, shrink=0.7, pad=0.02, label="[J m$^{-3}$]", ticks=mpl.ticker.MaxNLocator(4)))
# h = ax[5].contourf(ni_eke.time, ni_eke.z, ni_eke.data, extend="both", levels=np.arange(0.25, 4, 0.25), cmap="Spectral_r")
gv.plot.contourf_hide_edges(h)
# plt.colorbar(h, ax=ax[5])
# ni_eke.plot(ax=ax[5], extend="both", vmax=4, cmap="Spectral_r", cbar_kwargs=dict(aspect=30, shrink=0.7))
ax[5] = mld.gv.tcoarsen(n=30*24*2).gv.tplot(ax=ax[5], color="k", linewidth=1, alpha=0.5)
ax[5].invert_yaxis()
ax[5].set(xlabel="", ylabel="depth [m]")
ax[5].annotate("NI KE", xy=(0.95, 0.03), xycoords="axes fraction", backgroundcolor="w", ha="right")

h = td.plot.contourf(ax=ax[6], cmap="Spectral_r", levels=np.arange(3.5, 13.5, 0.5), cbar_kwargs=dict(label="[°C]", aspect=30, shrink=0.7, pad=0.02, ticks=mpl.ticker.MaxNLocator(4)))
gv.plot.contourf_hide_edges(h)
ax[6] = mld.gv.tcoarsen(n=30*24*2).gv.tplot(ax=ax[6], color="k", linewidth=1, alpha=0.5)
ax[6].set(xlim=[np.datetime64("2019-05-01"), np.datetime64("2020-10-10")])
ax[6].invert_yaxis()
ax[6].set(xlabel="", ylabel="depth [m]")
ax[6].annotate("temperature", xy=(0.95, 0.03), xycoords="axes fraction", backgroundcolor="w", ha="right")

gv.plot.subplotlabel(ax, x=0.025, y=0.88)

for axi in ax:
    gv.plot.axstyle(axi)
    axi.set(title="")
gv.plot.concise_date_all()

plot_fig = False
if plot_fig:
    name = "surface_forcing"
    niskine.io.png(name)
    niskine.io.pdf(name)

# %%
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 5),
                       constrained_layout=True, sharex=True,
                       height_ratios=[1, 1, 1, ])

velopts = dict(vmin=-0.75, vmax=0.75, rasterized=True, yincrease=False, cmap="RdBu_r")

adcp.u.plot(ax=ax[0], cbar_kwargs=dict(aspect=30, shrink=0.7, pad=0.02, label="[m$\,$s$^{-1}$]", ticks=mpl.ticker.MaxNLocator(4)), **velopts)
ax[0].annotate("eastward velocity", xy=(0.95, 0.03), xycoords="axes fraction", backgroundcolor="w", ha="right")
adcp.v.plot(ax=ax[1], cbar_kwargs=dict(aspect=30, shrink=0.7, pad=0.02, label="[m$\,$s$^{-1}$]", ticks=mpl.ticker.MaxNLocator(4)), **velopts)
ax[1].annotate("northward velocity", xy=(0.95, 0.03), xycoords="axes fraction", backgroundcolor="w", ha="right")

h = td.plot.contourf(ax=ax[2], cmap="Spectral_r", levels=np.arange(3.5, 13.5, 0.5), cbar_kwargs=dict(label="[°C]", aspect=30, shrink=0.7, pad=0.02, ticks=mpl.ticker.MaxNLocator(4)))
gv.plot.contourf_hide_edges(h)
# ax[2] = mld.gv.tcoarsen(n=30*24*2).gv.tplot(ax=ax[2], color="w", linewidth=1.75, alpha=1)
# ax[2] = mld.gv.tcoarsen(n=30*24*2).gv.tplot(ax=ax[2], color="k", linewidth=1, alpha=1)
ax[2].set(xlim=[np.datetime64("2019-05-01"), np.datetime64("2020-10-10")])
ax[2].invert_yaxis()
ax[2].set(xlabel="", ylabel="depth [m]")
ax[2].annotate("temperature", xy=(0.95, 0.03), xycoords="axes fraction", backgroundcolor="w", ha="right")

gv.plot.subplotlabel(ax, x=0.01, y=0.88)

for axi in ax:
    gv.plot.axstyle(axi)
    axi.set(title="", xlabel="")
gv.plot.concise_date(ax[-1])

plot_fig = True
if plot_fig:
    name = "vel_temp"
    niskine.io.png(name)
    niskine.io.pdf(name)

# %%
