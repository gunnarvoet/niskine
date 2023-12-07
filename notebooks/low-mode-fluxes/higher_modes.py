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
# # NISKINe higher NI modes

# %% [markdown]
# ## Mooring data structures

# %% [markdown]
# For the NISKINe mooring, add zero NI velocity close to the bottom to constrain the mode fits.

# %%
m1 = niskine.mooring.NISKINeMooring(add_bottom_adcp=True, add_bottom_zero=True)

# %%
m1

# %% [markdown]
# ## Calculate fluxes

# %%
N = niskine.flux.Flux(mooring=m1, bandwidth=1.06, runall=True, climatology="ARGO", nmodes=6, adcp_vertical_resolution=50)

# %%
N.find_modes()

# %%
cn = N.modes.mean(dim="time", keep_attrs=True).eigenspeed

# %%
fig, ax = gv.plot.quickfig(grid=True, h=3)
cn.plot();

# %%
f = gv.ocean.inertial_frequency(lat=N.mooring.lat)

# %% [markdown]
# $$\omega_n^2 - f^2 = c_n^2 \mathbf{K}^2$$

# %%
f**2

# %%
k = (2*np.pi)/(200e3)
fig, ax = gv.plot.quickfig()
(np.sqrt(cn**2*k**2 + f**2) / f).plot()
ax.set(ylabel="$\omega_n$")

# %%
om_n = 1.055*f
k = np.sqrt((om_n**2 - f**2) / cn[0]**2)

# %%
N.flux.fx_ni.gv.plot(robust=True)

# %%
niskine.flux.plot_up_one_time_step(N, ti=200, nmodes=6)
niskine.io.png("mode_fits_6_modes")

# %%
niskine.flux.plot_both_mode_fits_one_time_step(N, ti=100)

# %%
N.up.sel(mode=4).gv.plot(cmap="RdBu_r")

# %%
N.up.sel(mode=5).gv.plot(cmap="RdBu_r")

# %%
N.up.sel(mode=6).gv.plot(cmap="RdBu_r")

# %% [markdown]
# ## Calculate average flux magnitudes and seasonal mean

# %%
Nmag, Ndir = niskine.flux.flux_mag_and_dir(N)
print(f"{Nmag.mean().item():1.1f} {Nmag.units}")

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
