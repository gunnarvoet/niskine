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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
import gsw
import scipy

import gvpy as gv

import niskine

# %reload_ext autoreload
# %autoreload 2

plt.ion()

# %config InlineBackend.figure_format = 'retina'

# %%
cfg = niskine.io.load_config()

# %%
gv.plot.helvetica()
mpl.rcParams["lines.linewidth"] = 1

# %% [markdown]
# # NISKINe low-mode NI fluxes (final)
# This is the final version!!
#
# I found an error in the mode fitting procedure for $\eta$ which has been corrected now.
#
# I had a final look at OSNAP MM3 but the velocity coverage is not sufficient for mode fits.

# %% [markdown]
# ## Mooring data structures

# %% [markdown]
# For the NISKINe mooring, add zero NI velocity close to the bottom to constrain the mode fits.

# %%
m1 = niskine.mooring.NISKINeMooring(add_bottom_adcp=True, add_bottom_zero=True)

# %%
m1

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "f9337f7b8cb73", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
os4 = niskine.mooring.OSNAPMooring(moorstr='UMM4')

# %% [markdown]
# ## Calculate fluxes

# %%
N = niskine.flux.Flux(mooring=m1, bandwidth=1.06, runall=True, climatology="ARGO")

# %%
O = niskine.flux.Flux(mooring=os4, bandwidth=1.06, runall=True, climatology="ARGO")

# %% [markdown]
# ## Calculate average flux magnitudes and seasonal mean

# %%
Nmag, Ndir = niskine.flux.flux_mag_and_dir(N)
print(f"{Nmag.mean().item():1.1f} {Nmag.units}")

# %%
print(f"highest 5% are in the range {np.percentile(Nmag, 95):1.1f} {Nmag.units} - {np.max(Nmag).item():1.1f} {Nmag.units}")

# %%
Omag, Odir = niskine.flux.flux_mag_and_dir(O)
print(f"{Omag.mean().item():1.1f} {Omag.units}")

# %%
print(f"highest 5% are in the range {np.percentile(Omag, 95):1.1f} {Omag.units} - {np.max(Omag).item():1.1f} {Omag.units}")

# %%
np.percentile(Omag, 95)

# %%
Nmag.plot()

# %%
Omag.plot()

# %% [markdown]
# ## Plot mode fit examples

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_both_mode_fits_one_time_step(N, ti=6100)
niskine.io.png("vel_mode_fit_example")
niskine.io.pdf("vel_mode_fit_example")

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_up_one_time_step(N, ti=1500)
# niskine.io.png("vel_mode_fit_example")

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_eta_modes_one_time_step(N, ti=1500)

# %% [markdown]
# OSNAP

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_up_one_time_step(O, ti=3500)
# niskine.io.png("vel_mode_fit_example")

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_eta_modes_one_time_step(O, ti=3500)

# %%

# %% [markdown]
# ## Plot flux time series and seasonal mean

# %%
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True)
niskine.flux.plot_flux_time_series(ax[0], N, add_title=False)
niskine.flux.plot_flux_time_series(ax[1], O, add_legend=False, add_title=False)
for axi in ax:
    gv.plot.axstyle(axi)
    gv.plot.concise_date(axi)
ax[0].set(ylim=(-0.6, 0.2))
# ax[0].set(ylim=(-1.4, 0.9))
ax[1].set(ylim=(-1.4, 0.9))

gv.plot.annotate_upper_left("NISKINE M1", ax=ax[0])
gv.plot.annotate_upper_left("OSNAP MM4", ax=ax[1])

ax[0].set(xlim=(np.datetime64("2019-05-01"), np.datetime64("2021-06-01")))
ax[1].set(xlim=(np.datetime64("2014-05-01"), np.datetime64("2016-06-01")))

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
from matplotlib.gridspec import GridSpec
fig = plt.figure(layout="constrained", figsize=(9, 5))

gs = GridSpec(2, 7, figure=fig)
ax1 = fig.add_subplot(gs[0, :-2])
pax1 = fig.add_subplot(gs[0, -2:], projection="polar")
ax2 = fig.add_subplot(gs[1, :-2])
pax2 = fig.add_subplot(gs[1, -2:], projection="polar")
ax = [ax1, ax2]

# fig, axx = plt.subplots(nrows=2, ncols=2, width_ratios=[3, 1], figsize=(8, 5),
#                        constrained_layout=True)
# ax = axx[:, 0]
# pax = axx[:, 1]
niskine.flux.plot_flux_time_series(ax[0], N, add_title=False)
niskine.flux.plot_flux_time_series(ax[1], O, add_legend=False, add_title=False)
for axi in ax:
    gv.plot.axstyle(axi)
    gv.plot.concise_date(axi)
ax[0].set(ylim=(-0.6, 0.2))
# ax[0].set(ylim=(-1.4, 0.9))
ax[1].set(ylim=(-1.4, 0.9))

gv.plot.annotate_upper_left("NISKINE M1", ax=ax[0])
gv.plot.annotate_upper_left("OSNAP MM4", ax=ax[1])

ax[0].set(xlim=(np.datetime64("2019-05-01"), np.datetime64("2021-06-01")))
ax[1].set(xlim=(np.datetime64("2014-05-01"), np.datetime64("2016-06-01")))


niskine.flux.plot_flux_polar(pax1, N)
niskine.flux.plot_flux_polar(pax2, O)

niskine.io.png("low_mode_fluxes")
niskine.io.pdf("low_mode_fluxes")
