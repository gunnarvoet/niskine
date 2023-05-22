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
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm.notebook import tqdm

import gvpy as gv
# import osnap
import niskine

# %reload_ext autoreload
# %autoreload 2

# %config InlineBackend.figure_format = 'retina'

# %%
cfg = niskine.io.load_config()

# %% [markdown]
# # NISKINe Wind Forcing

# %% [markdown]
# ## Updates

# %% [markdown]
# **Update 18 Aug 2022**: I tried to calculate the NI wind work with velocities averaged over the mixed layer only but have failed so far. For some reason the result has a flipped sign when compared to what I had earlier. I also noticed that I had not bandpass filtered the wind stress around near-inertial time scales.
#
# Turns out I was not averaging over the extracted mixed layer depth data but over the full profiles. Things look much better now and I have also corrected the error with the unfiltered wind stress.

# %% [markdown]
# **Update 14 Sep 2022:** Moved mixed layer depth conversion from mat to netcdf and mixed layer depth average velocity calculation to the `niskine` libraries outside of this notebook.

# %% [markdown]
# **Update 20 Sep 2022:** Started an abstract base class for the wind work calculation that we can use for both NISKINe and OSNAP.

# %% [markdown]
# **Update 21 Sep 2022:** Cleaned up the NISKINe wind work calculation. Found that the mixed layer average velocity had a bunch of NaNs, interpolating over them increases the cumulative wind work by about 30%. Merging then wind work calculation into the main repository.

# %% [markdown]
# **Update 22 Sep 2022:** Added OSNAP wind work based on ocean surface velocity averaged over a constant depth layer.

# %% [markdown]
# ## Code Structure

# %% [markdown]
# We want to calculate wind work for a NISKINe and OSNAP moorings.
# - Wind data is the same (hourly ERA5). Interpolate to whatever temporal resolution the ocean surface velocity has. Actually, right now I interpolate the ocean velocity to the wind data to achieve hourly resolution.
# - Ocean surface velocity estimates may differ between the calculations, they need to implented in the actual classes for each mooring type.
# - Make the current coupling parameter ($s_w=0.3$) as used in Vic et al (2021) an optional parameter.
#
# What we really want is a function that lets us calculate the NI wind work at any mooring, whether it is NISKINe or OSNAP. This is now happening in this notebook. I constructed an abstract base class for the wind work calculation, and then make specific cases for OSNAP and NISKINe. The mooring types between these projects differ a bit, but by using the abstract base class we can get them to a similar format and can run the same analysis functions on them.

# %% [markdown]
# ## ERA5 wind data

# %%
wind = xr.open_dataset(cfg.data.wind.era5)

# %% [markdown]
# ## Approach

# %% [markdown]
# [@vicetal21](file:///Users/gunnar/Projects/gvbib/articles/Vic/Vic2021.pdf) have a good description of their approach for calculating wind work from ERA5. The wind work  on near-inertial motions is
# $$
# W_{NI} = \mathbf{\tau_{NI}} \cdot \mathbf{u}_{NI}
# $$
# with wind stress and surface velocity in the near-inertial band. Wind stress is calculated via
# $$\mathbf{\tau} = \rho_a C_D \mathbf{u_r} |\mathbf{u_r}|$$
# with air density $\rho_a$, drag coefficient $C_D$, and relative wind speed $\mathbf{u_r} = \mathbf{u_{10}} - \mathbf{u_s}$. Here, $\mathbf{u_{10}}$ is the ERA5 wind speed (at 10m above sea level) and $\mathbf{u_s}$ is the surface velocity in the ocean. A correction factor is applied to the calculation of $\mathbf{u_r}$ to account for current feedback on the wind stress in reanalysis products [[@renaultetal2020](file:///Users/gunnar/Projects/gvbib/articles/Renault/Renault2020.pdf)]: $\mathbf{u_r} = \mathbf{u_{10}} - (1-s_w) \mathbf{u_s}$ where $s_w=0.3$ is the globally averaged value of the current-wind coupling coefficient.
#
# [@klenzetal22](file:///Users/gunnar/Projects/gvbib/articles/Klenz/Klenz2022.pdf) compare ERA5-based wind work estimates to direct wind & surface velocity measurements from Minimets deployed during NISKINe. They  do not apply the coefficient for current-wind coupling as [@vicetal21](http://dx.doi.org/10.1175%2Fjpo-d-20-0097.1). I asked Thilo Klenz in an email, he said he was not aware of the correction factor but believes that it does not make a huge difference since ocean velocities are small compared to wind velocities.
#
# Following Vic et al and others, we calculate near-inertial wind work (or the near-inertial air-sea energy flux) $\Pi$ as
# $$
# \Pi = \vec{\tau_\mathrm{f}} \cdot \vec{u}_\mathrm{f}
# $$
# where $\vec{u}_\mathrm{f}$ denotes the inertial mixed layer currents and $\tau_\mathrm{f}$ near-inertial band wind stress. We calculate $ \vec{u}_\mathrm{f}$ as an average velocity over the mixed layer and then bandpass the velocity time series around f (with a bandwidth factor of 1.05). Wind stress $\tau$ is calculated from the relative wind velocity.

# %% [markdown]
# ## NI Wind Work

# %% [markdown]
# Calculate near-inertial wind work for the NISKINe mooring M1. Let's see what difference the current feedback correction factor makes.

# %%
N = niskine.calcs.NIWindWorkNiskine()
Nc = niskine.calcs.NIWindWorkNiskine(apply_current_feedback_correction=False)

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4),
                       constrained_layout=True)
N.wind_work_int.plot(color='C0', linewidth=1, label='current feedback correction')
Nc.wind_work_int.plot(color='C4', linewidth=1, label='no correction')
ax.legend()
ax.set(title='cumulative NI wind work at NISKINe M1', ylabel="P$_\mathregular{W}$ [kJ/m$^2$]")
gv.plot.axstyle(ax)
gv.plot.concise_date(ax)
niskine.io.png('cumulative_wind_work_niskine_m1', subdir='wind-work')
niskine.io.pdf('cumulative_wind_work_niskine_m1', subdir='wind-work')

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4),
                       constrained_layout=True)
N.wind_work_int.plot(color='C0', linewidth=1, label='current feedback correction')
Nc.wind_work_int.plot(color='C4', linewidth=1, label='no correction')
ax.legend()
ax.set(title='cumulative NI wind work at NISKINe M1', ylabel="P$_\mathregular{W}$ [kJ/m$^2$]")
gv.plot.axstyle(ax)
gv.plot.concise_date(ax)
niskine.io.png('cumulative_wind_work_niskine_m1', subdir='wind-work')
niskine.io.pdf('cumulative_wind_work_niskine_m1', subdir='wind-work')

# %% [markdown]
# Plot ocean velocity, wind stress, and air-sea energy flux for one component

# %%
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(7, 5),
                       constrained_layout=True, sharex=True)
N.v_ocean_ni.plot(ax=ax[0], label=r'v$_f$', color='0.2')
ax[0].set(ylabel='v$_{\mathregular{f}}$ [m/s]')
gv.plot.annotate_corner('near-inertial meridional velocity', ax[0], quadrant=4, fw='normal')
N.tauy_ni.plot(ax=ax[1], color='0.2')
ax[1].set(ylabel=r'$\tau_y$ [N/m$^2$]')
gv.plot.annotate_corner('near-inertial meridional wind stress', ax[1], quadrant=4, fw='normal')
N.wind_work.plot(ax=ax[2], color='0.2');
ax[2].set(ylabel='$\Pi$ [W/m$^2$]')
gv.plot.annotate_corner('near-inertial air-sea energy flux', ax[2], quadrant=4, fw='normal')
for axi in ax:
    axi.set(title='', xlabel='')
    gv.plot.axstyle(axi, grid=False, ticks='in', fontsize=9)
gv.plot.concise_date(ax[2])
niskine.io.png('wind_work_niskine_m1', subdir='wind-work')
niskine.io.pdf('wind_work_niskine_m1', subdir='wind-work')

# %% [markdown]
# Save Results

# %%
cfg.data.wind_work.niskine_m1.as_posix()

# %%
N.wind_work.to_netcdf(cfg.data.wind_work.niskine_m1.as_posix())

# %%
N.wind_work_int.to_netcdf(cfg.data.wind_work.niskine_m1_cumulative.as_posix())

# %% [markdown]
# ## OSNAP Wind Work

# %% [markdown]
# For surface velocities we are currently using an average over the upper 200m current observations. We could possibly refine this using the Argo mixed layer climatology.

# %%
No3 = niskine.calcs.NIWindWorkOsnap(mooring=3)

# %%
No4 = niskine.calcs.NIWindWorkOsnap(mooring=4)

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.5), constrained_layout=True)
No3.wind_work_int.plot(ax=ax, color="C0", linewidth=1, label="OSNAP UMM3")
No4.wind_work_int.plot(ax=ax, color="C4", linewidth=1, label="OSNAP UMM4")
ax.set(
    ylabel="P$_\mathregular{W}$ [kJ/m$^2$]",
    xlabel="",
    title="OSNAP cumulative NI wind work",
)
gv.plot.axstyle(ax, grid=True, ticks="in")
gv.plot.concise_date(ax)
ax.legend()
niskine.io.png("osnap_wind_work", subdir="wind-work")
niskine.io.pdf("osnap_wind_work", subdir="wind-work")

# %%
No3.wind_work.to_netcdf(cfg.data.wind_work.osnap_umm3)
No3.wind_work_int.to_netcdf(cfg.data.wind_work.osnap_umm3_cumulative)
No4.wind_work.to_netcdf(cfg.data.wind_work.osnap_umm4)
No4.wind_work_int.to_netcdf(cfg.data.wind_work.osnap_umm4_cumulative)
