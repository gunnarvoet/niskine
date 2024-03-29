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

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "80aa11a68a82c8", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
# %matplotlib inline
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import gsw

import gvpy as gv
import niskine

# %reload_ext autoreload
# %autoreload 2
# %autosave 0

# %config InlineBackend.figure_format = 'retina'

# %%
cfg = niskine.io.load_config()

# %%
mld = xr.open_dataarray(cfg.data.ml.mld)

# %% [markdown]
# # Compare wind work and NI EKE

# %%
# load NI EKE
a = xr.open_dataset(cfg.data.ni_eke_m1)

# %%
# load wind work
wind_work = xr.open_dataarray(cfg.data.wind_work.niskine_m1)
wind_work.close()
wind_work_c = xr.open_dataarray(cfg.data.wind_work.niskine_m1_cumulative)
wind_work_c.close()
wind_stress = xr.open_dataarray(cfg.data.wind_work.niskine_m1_wind_stress)
wind_stress.close()

# load MLD
mld = xr.open_dataarray(cfg.data.ml.mld)
mld.close()

# %%
ni_eke = a.ni_eke
mldi = mld.interp_like(ni_eke)
mld_mask = ni_eke.z < mldi

deep_mask = (ni_eke.z > 500) & (ni_eke.z < 1200)

# %%
int_all = ni_eke.where(ni_eke.z<1300).sum(dim="z") * 16 / 1e3
int_alls = int_all.rolling(time=100).mean()
int_mld = ni_eke.where(mld_mask).sum(dim="z") * 16 / 1e3
int_mlds = int_mld.rolling(time=100).mean()
int_deep = ni_eke.where(deep_mask).sum(dim="z") * 16 / 1e3
int_deeps = int_deep.rolling(time=100).mean()

# %%
fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(7.5, 4), constrained_layout=True, sharex=True
)

wind_work_c.plot(ax=ax[0], linewidth=1.5, color="0.2")
ax[0].set(ylabel="$\int \Pi_\mathrm{NI}\,\mathrm{dt}$ [kJ/m$^2$]", title="")
ax[0].annotate(
    "cumulative NI wind work",
    xy=(np.datetime64("2020-10-01"), 3.3),
    ha="right",
    color="0.2",
)

# (a.ni_eke.sum(dim="z") * 16 / 1e3).plot(ax=ax[1], linewidth=2, color="w")
# int_alls.plot(ax=ax[1], linewidth=1, color="k")
# ax[1].annotate(
#     "$\mathrm{EKE}_\mathrm{NI}\ 0 - 1500\,\mathrm{m}$",
#     xy=(np.datetime64("2020-01-01"), 1.8),
#     ha="right",
#     color="k",
# )

int_mlds.plot(ax=ax[1], linewidth=2, color="w")
int_mlds.plot(ax=ax[1], linewidth=1.0, color="C6")

int_deeps.plot(ax=ax[1], linewidth=2, color="w")
int_deeps.plot(ax=ax[1], linewidth=1.0, color="C0")

ax[1].annotate(
    "mixed layer",
    xy=(np.datetime64("2020-04-01"), 0.8),
    ha="left",
    color="C6",
)
ax[1].annotate(
    "500-1200m",
    xy=(np.datetime64("2019-06-01"), 0.6),
    ha="left",
    color="C0",
)

ax[1].set(ylabel="$\int \mathrm{EKE}_\mathrm{NI}\,\mathrm{dz}$ [kJ/m$^2$]", title="")
gv.plot.concise_date(ax[1])

for axi in ax:
    gv.plot.axstyle(axi, grid=True)

gv.plot.subplotlabel(ax, x=0.01, y=0.88)

niskine.io.png("integrated_ni_eke_and_cumulative_wind_work")
niskine.io.pdf("integrated_ni_eke_and_cumulative_wind_work")

# %%
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(7.5, 8),
                       constrained_layout=True, sharex=True)

wind_work.plot(ax=ax[0])
wind_work_c.plot(ax=ax[1])

(a.ni_eke.sum(dim="z") * 16 / 1e3).plot(ax=ax[2])
gv.plot.concise_date(ax[2])
ax[2].set(ylabel="$\mathrm{EKE}_\mathrm{NI}$ [kJ/m$^2$]")

(ni_eke.where(mld_mask).sum(dim="z") * 16 / 1e3).plot(ax=ax[3])
gv.plot.concise_date(ax[3])
ax[3].set(ylabel="$\mathrm{EKE}_\mathrm{NI} (ML)$ [kJ/m$^2$]")

(ni_eke.where(deep_mask).sum(dim="z") * 16 / 1e3).plot(ax=ax[4])
gv.plot.concise_date(ax[4])
ax[4].set(ylabel="$\mathrm{EKE}_\mathrm{NI} (deep)$ [kJ/m$^2$]")

for axi in ax:
    gv.plot.axstyle(axi, grid=True)
    axi.set(title=None)

# %%
