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

# %% [markdown]
# ## WOCE Climatology

# %% [markdown]
# Development code for `niskine.io.climatology_woce`. Input is lon/lat/bottom-depth so we can use it for all OSNAP and NISKINe moorings.

# %% [markdown]
# ### Read Mooring Data

# %% [markdown]
# We just need lon / lat / bottom depth to test the function.

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "f9337f7b8cb73", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
os4 = niskine.osnap.Mooring(moorstr='UMM4')

# %%
os4.lon

# %% [markdown]
# The goal is to have one DataArray with $N^2$ and coordinate `z`, in the case of using the Argo climatology also `time`. Also one DataArray with $T_z$.

# %% [markdown] janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "eeb9cb97ebc1e8", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
# ### WOCE Climatology

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "d40bfbc4d8dce", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
woce = gv.ocean.woce_climatology()
wprf2 = woce.interp(lon=os4.lon+360, lat=os4.lat).squeeze()
wprf = woce.sel(lon=os4.lon+360, lat=os4.lat, method='nearest').squeeze()

# %% [markdown]
# I think by interpolating we lose some information at the bottom as it only goes to the depth of the shallowest neighboring profiles. We could probably do better by extending bottom values or coming up with interpolation that ignores NaN's. For now we'll just pick the profile that is closest...

# %%
fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(6, 5), constrained_layout=True, sharey=True
)
wprf.t.plot(ax=ax[0], y="depth")
wprf2.t.plot(ax=ax[0], y="depth")
(wprf.t - wprf2.t).plot(ax=ax[1], y="depth")
ax[0].set(title="temperature", xlabel="temperature [$^{\circ}$C]")
ax[1].set(title="temperature difference\nbetween methods", ylabel="", xlabel="$\Delta$ Temp [$^{\circ}$C]")
[gv.plot.axstyle(axi) for axi in ax]
ax[0].invert_yaxis()

# %% [markdown]
# #### Calculate buoyancy frequency.

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "d40bfbc4d8dce", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
SA = gsw.SA_from_SP(wprf.s, wprf.p, wprf.lon, wprf.lat)
CT = gsw.CT_from_t(SA, wprf.t, wprf.p)
N2, pmid = gsw.Nsquared(SA, CT, wprf.p, lat=wprf.lat)
ni = np.isfinite(N2)
N2 = N2[ni]
N2z = gsw.z_from_p(pmid[ni], os4.lat)
# extend constant to bottom
if np.max(np.abs(N2z)) < os4.depth:
    N2z = np.append(N2z, np.abs(os4.depth)*-1)
    N2 = np.append(N2, N2[-1])
N2 = xr.DataArray(N2, coords={'z': (['z'], N2z)}, dims=['z'])

# %%
N2.z.shape

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "caef4b9196841", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
fig, ax = gv.plot.quickfig(w=3)
N2.plot(y='z', color='0.2');
ax.set(xlabel='N$^2$ [1/s$^2$]', ylabel='z [m]');

# %% [markdown]
# #### Calculate Temperature Gradient

# %% [markdown]
# Using `xarray.diferentiate` which uses second order accurate central differences:

# %%
Tz = wprf.th.differentiate('depth')

# %% [markdown]
# Simple differencing

# %%
Tz2 = np.diff(wprf.th) / np.diff(wprf.depth)
Tz2_z = wprf.depth[:-1] + np.diff(wprf.depth)/2

# %%
fig, ax = gv.plot.quickfig(w=3.5, yi=True)
Tz.plot(y='depth', color='0.2', ax=ax);
ax.plot(Tz2, Tz2_z);
ax.set(title='Temperature Gradient', xlabel='T$_z$ [°/m]');
ax.invert_yaxis()

# %% [markdown]
# The second order method looks a little bit cleaner. Extend it towards the bottom.

# %%
zi = np.flatnonzero((Tz.depth >= -3000) & (np.isnan(Tz.data)))
deeptz = Tz.where(((Tz.depth>=3000) & (Tz.depth<2000))).mean()
Tz[zi] = deeptz

fig, ax = gv.plot.quickfig(w=3.5, yi=True)
Tz.plot(y='depth', color='0.2', ax=ax);
ax.set(title='Temperature Gradient', xlabel='T$_z$ [°/m]');
ax.invert_yaxis()

# %% [markdown]
# #### Compare mean mooring data to climatology
# The mean values are not quite that exact as the instruments move in the vertical, thus we average over a certain depth range here.

# %%
mctd = os4.ctd.mean(dim='time')

# %%
fig, ax = gv.plot.quickfig(w=3.5, yi=False)
pstd = os4.ctd.p.std(dim='time')
(2*pstd).plot(y='nomz', color='0.2');
ax.set(xlabel=r'$2\sigma_p$ [dbar]')
ax.set_title('CTD Pressure Standard Deviation', fontsize=13);

# %%
fig, ax = gv.plot.quickfig(w=3.5, yi=False)
ax.plot(wprf.t, wprf.p, color='0.2', label='WOCE')
ax.plot(mctd.t, mctd.p, marker='o', linestyle='', color='0.2', label='OSNAP')
ax.legend();
ax.set(ylabel='pressure [dbar]', title='Mean Temperature');

# %% [markdown]
# ### Function

# %% [markdown]
# Everything developed above now lives in a lil function at `niskine.io.climatology_woce`. Input is lon/lat/bottom-depth so we can use it for all OSNAP and NISKINe moorings.

# %%
N2, Tz = niskine.clim.climatology_woce(os4.lon, os4.lat, os4.depth)

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 5),
                       constrained_layout=True, sharey=True)
N2.plot(ax=ax[0], y='depth')
Tz.plot(ax=ax[1], y='depth', yincrease=False)
for axi in ax:
    axi.set(title='')
    gv.plot.axstyle(axi)
ax[1].set(ylabel='');
