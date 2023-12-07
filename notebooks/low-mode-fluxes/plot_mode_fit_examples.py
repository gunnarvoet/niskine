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
# # NISKINe plot mode fit examples

# %% [markdown]
# Mooring data structures
#
# For the NISKINe mooring, add zero NI velocity close to the bottom to constrain the mode fits.

# %%
m1 = niskine.mooring.NISKINeMooring(add_bottom_adcp=True, add_bottom_zero=True)

# %%
m1

# %% [markdown]
# Calculate fluxes

# %%
N = niskine.flux.Flux(mooring=m1, bandwidth=1.06, runall=True, climatology="ARGO")

# %% [markdown]
# Plot mode fit examples

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_both_mode_fits_one_time_step(N, ti=6100)
niskine.io.png("vel_mode_fit_example")
niskine.io.pdf("vel_mode_fit_example")

# %% [markdown]
# ## Energy Budget

# %% [markdown]
# Some notes on Sam Kelly's comments. Sam suggests to calculate mode-1 wind-work as
#
# $$
# \Pi_1 = \tau_0 u_1 \phi_1
# $$
#
# He suggests dividing energy by wind-work to get a characteristic decay timescale $T$. How does this work in units?
#
# Wind-work is units of W/m$^2$ which is also a flux. This may also be called power input per unit area and also has units of J/s/m$^2$.
#
# Energy is power integrated over time or J/m$^2$. This makes sense - our plot of NI EKE is in units of J/m$^3$ since it has not been depth-integrated. Thus dividing energy by wind-work will yield time.
#
# Alford & Whitmont (2007) calculate a replenishment time scale similar to what Sam suggests. They write (note that their depth-integral is near-inertial kinetic energy is the product of water depth and depth-averaged NI kinetic energy):
# > The ratio of depth-integrated $\mathrm{KE}_\mathrm{moor}$ to the energy flux
# $\mathrm{F}_\mathrm{ML}$ gives a replenishment time scale,
# $$
# \tau = \frac{\rho\, D\, \langle \mathrm{KE}_\mathrm{moor} \rangle}{\mathrm{F}_\mathrm{ML}}
# $$
# which can be interpreted as the time it would take to substantially reduce $\mathrm{KE}_\mathrm{moor}$ (a factor of $e^{-1}$) if forcing stopped.
#
#
#

# %%

# %%

# %% [markdown]
# Compare shallow vs deep mixed layer with synthetic data.

# %%
modes = N.modes.isel(time=6100)

# %%
modes.hmodes.plot(hue="mode", y="z", yincrease=False);

# %%
modes.hmodes.shape

# %%
modes.z.shape

# %%
data0 = np.concatenate([np.ones((1, 3)) * 0.1, np.zeros((1, 52))], axis=1)[0]
z0 = np.arange(20, (len(data0)) * 20, 20)
z0 = np.append(z0, 2800)

# %%
data1 = np.concatenate([np.ones((1, 20)) * 0.1, np.zeros((1, 35))], axis=1)[0]
z1 = np.arange(20, (len(data1)) * 20, 20)
z1 = np.append(z1, 2800)

# %%
beta0 = niskine.flux.project_on_vmodes(data0, z0, modes.hmodes.data, modes.z.data)
beta1 = niskine.flux.project_on_vmodes(data1, z1, modes.hmodes.data, modes.z.data)

# %%
mode1_0 = np.multiply(modes.hmodes.sel(mode=1), beta0[0]) 

# %%
mode1_1 = np.multiply(modes.hmodes.sel(mode=1), beta1[0]) 

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 5),
                       constrained_layout=True, sharey=True)

mode1_0.plot(ax=ax[1], y="z", yincrease=False, color="r")
mode1_1.plot(ax=ax[1], y="z", yincrease=False, color="b")

ax[0].plot(data1, z1, "b-", marker=".", alpha=0.7)
ax[0].plot(data0, z0, "r-", marker=".", alpha=0.7)
for axi in ax:
    axi.set(title=None)
ax[0].set(xlabel="u [m/s]", ylabel="depth [m]", ylim=(3.1e3, -100))
ax[1].set(ylabel="")
for axi in ax:
    gv.plot.axstyle(axi, grid=True)
niskine.io.png("mode1_examples", subdir="low-mode-fluxes")

# %%
