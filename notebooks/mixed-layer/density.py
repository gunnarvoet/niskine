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
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm.notebook import tqdm
import gsw

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
# Load mooring config

# %%
m = niskine.strat.MooringConfig()

# %%
m.microcat

# %% [markdown]
# # NISKINe T-S Relationship

# %% [markdown]
# Load microcat data

# %%
mmc = niskine.io.read_microcats()

# %% [markdown]
# Let's look at T-S in the shallow microcat

# %%
ma = mmc.sel(sn=4923)
ma.attrs["nomz"] = np.int32(np.round(m.info(4923).depth))
mb = mmc.sel(sn=2864)
mb.attrs["nomz"] = np.int32(np.round(m.info(2864).depth))
mc = mmc.sel(sn=12710)
mc.attrs["nomz"] = np.int32(np.round(m.info(12710).depth))
md = mmc.sel(sn=12711)
md.attrs["nomz"] = np.int32(np.round(m.info(12711).depth))
me = mmc.sel(sn=12712)
me.attrs["nomz"] = np.int32(np.round(m.info(12712).depth))

# %%
mint = 2
maxt = 14

mins = 34.5
maxs = 35.8

tspace = np.linspace(mint, maxt, 200)
sspace = np.linspace(mins, maxs, 200)

Tg, Sg = np.meshgrid(tspace, sspace)
sigma_theta = gsw.sigma0(Sg, Tg)
cnt = np.linspace(sigma_theta.min(), sigma_theta.max(), 200)

fig, ax = plt.subplots(figsize=(2, 2))
cs = ax.contour(Sg, Tg, sigma_theta, colors="grey", zorder=1)
cl = plt.clabel(cs, fontsize=8, inline=True, fmt="%.2f")


# %%
def density_contours(ax, fontsize=8):
    # xaxis is SA
    mins, maxs = ax.get_xlim()
    
    # yaxis is CT
    mint, maxt = ax.get_ylim()

    tspace = np.linspace(mint, maxt, 200)
    sspace = np.linspace(mins, maxs, 200)

    Tg, Sg = np.meshgrid(tspace, sspace)
    sigma_theta = gsw.sigma0(Sg, Tg)
    cnt = np.linspace(sigma_theta.min(), sigma_theta.max(), 200)

    cs = ax.contour(Sg, Tg, sigma_theta, 12, colors="grey", linewidths=0.5, zorder=1)
    cl = plt.clabel(cs, fontsize=fontsize, inline=True, fmt="%.2f")


# %%
fig, ax = gv.plot.quickfig(fgs=(4,4))

ax.set(xlim=[34.8, 35.6])

# cs = ax.contour(Sg, Tg, sigma_theta, 20, colors="grey", linewidths=0.5)
# cl = plt.clabel(
#     cs,
#     fontsize=8,
#     inline=True,
#     fmt="%.1f",
#     manual=[(35.5, 4), (35.5, 12), (35.5, 8), (35.5, 10)],
# )
opts = dict(
    marker=".",
    ms=2,
    linestyle="",
    alpha=0.05,
)
ax.plot(me.SA, me.CT, label=me.nomz, **opts)
ax.plot(md.SA, md.CT, label=md.nomz, **opts)
ax.plot(mc.SA, mc.CT, label=mc.nomz, **opts)
ax.plot(mb.SA, mb.CT, label=mb.nomz, **opts)
ax.plot(ma.SA, ma.CT, label=ma.nomz, **opts)
density_contours(ax, fontsize=6)
# ax.legend()
ax.set(xlabel="salinity [g/kg]", ylabel="temperature [°C]", title="NISKINe M1 SBE37 data")
niskine.io.png("ts_all_microcat", subdir="microcat")

# %%
fig, ax = gv.plot.quickfig()
ax.set(xlim=[35, 35.4], ylim=[4, 10])
cs = ax.contour(Sg, Tg, sigma_theta, 30, colors="grey", linewidths=0.5)
cl = plt.clabel(
    cs,
    fontsize=8,
    inline=True,
    fmt="%.1f",
    manual=[(35.5, 4), (35.5, 12), (35.5, 8), (35.5, 10)],
)
opts = dict(marker='.', ms=2, linestyle="", alpha=0.05, )
ax.plot(md.SA, md.CT, **opts)
ax.plot(mc.SA, mc.CT, **opts)
# ax.plot(mb.SA, mb.t, **opts)
# ax.plot(ma.SA, ma.t, **opts)

# %%
fig, ax = gv.plot.quickfig(fgs=(4, 4))
ax.set(xlim=[35.12, 35.14], ylim=[2.1, 2.7])
opts = dict(marker='.', ms=2, linestyle="", alpha=0.1, )
ax.plot(me.SA, me.CT, **opts)
density_contours(ax)


# %%
timesel = "2019"
opts = dict(alpha=0.4, density=False)
ma.sel(time=timesel).sg0.plot.hist(bins=100, **opts);
mb.sel(time=timesel).sg0.plot.hist(bins=100, **opts);
mc.sel(time=timesel).sg0.plot.hist(bins=100, **opts);
md.sel(time=timesel).sg0.plot.hist(bins=100, **opts);

# %%
me.sg0.plot.hist(bins=100);

# %% [markdown]
# Plot some short time series to visualize resolution.

# %%
timeslice = slice("2019-06-15", "2019-06-19")

# %%
ax = ma.SP.sel(time=timeslice).gv.tplot()
ax.set(title=f"NISKINe M1 SBE37 #{ma.sn.data}", ylabel="salinity [g/kg]")
niskine.io.png(f"sn{ma.sn.data}_example_time_series", subdir="microcat")

# %%
ax = me.SP.sel(time=timeslice).gv.tplot()
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=0.001))
ax.set(title=f"NISKINe M1 SBE37 #{me.sn.data}", ylabel="salinity [g/kg]")
niskine.io.png(f"sn{me.sn.data}_example_time_series", subdir="microcat")

# %% [markdown]
# Load depth-gridded temperature

# %%
temp = xr.open_dataarray(cfg.data.gridded.temperature)

# %%
ax = temp.sel(time="2019-06").gv.tplot()
ax.set(ylim=[600, 0])
