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
import pandas as pd
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

# %% [markdown]
# ## WOCE Argo Climatology

# %% [markdown]
# Development code for `niskine.clim.climatology_argo_woce`. Input is lon/lat/bottom-depth so we can use it for all OSNAP and NISKINe moorings.

# %% [markdown]
# ### Read Mooring Data
#
# We just need lon / lat / bottom depth to test the function.

# %%
lon, lat, bottom_depth = niskine.io.mooring_location(mooring=1)

# %%
argo = gv.ocean.woce_argo_profile(lon, lat)

# %%
argo

# %% [markdown]
# ## Climatological Data

# %% [markdown]
# The goal is to have one DataArray with $N^2$ and coordinate `z`, in the case of using the Argo climatology also `time`. Also one DataArray with $T_z$.

# %% [markdown] janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "76a8c910556a48", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
# ### WOCE ARGO Climatology

# %% [markdown] janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "bfadcf6e96a33", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
# We can either interpolate to the correct location or pick the nearest neighbor. Again, interpolation cuts the profile short at depth. Until we have a better interpolation algorithm, let's just pick the nearest profile.

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "6fee65984b2e18", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
# %%time
if 0:
    argo = gv.ocean.woce_argo_profile(lon, lat, interp=True)
else:
    argo = gv.ocean.woce_argo_profile(lon, lat)

# %%
argo.coords['month'] = (['time'], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "57182d0a8081f8", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
fig, ax = gv.plot.quickfig(w=4, yi=False)
h = argo.t.plot(
    hue="time", y="z", ax=ax, add_legend=False
)

# %% [markdown]
# #### Calculate buoyancy frequency.

# %%
# calculate pressure from depth
argo['p'] = (['z'], gsw.p_from_z(-argo.z, lat).data)
argo = argo.transpose('z', 'time')

# %%
# bring pressure to same dimensions
_, argo['p'] = xr.broadcast(argo.s, argo.p)
# calculate absolute salinity
argo['SA'] = (['z', 'time'], gsw.SA_from_SP(argo.s, argo.p, argo.lon, argo.lat).data)
# calculate conservative temperature
argo['CT'] = (['z', 'time'], gsw.CT_from_t(argo.SA, argo.t, argo.p).data)

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "d40bfbc4d8dce", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
N2, pmid = gsw.Nsquared(argo.SA, argo.CT, argo.p, lat=lat)
N2z = -gsw.z_from_p(pmid, lat=lat)
N2 = xr.DataArray(N2, coords={'z': N2z[:,0], 'time': argo.time}, dims=['z', 'time'])
N2 = N2.where(np.isfinite(N2), drop=True)

# %% [markdown]
# There are a few negative values for $N^2$ in the upper ocean:

# %%
fig, ax = gv.plot.quickfig(yi=False, h=2)
N2.where(N2<0, drop=True).plot(yincrease=False)

# %% [markdown]
# Try with a sorted density profile.

# %%
argo['sg0'] = (['z', 'time'], gsw.sigma0(argo.SA, argo.CT).data)

# %%
# N2s = xr.full_like(argo.t, np.nan).rename('N2')
N2s = np.zeros((argo.t.shape[0]-1, argo.t.shape[1]))*np.nan
N2s = xr.DataArray(data=N2s, dims=['z', 'time'])
N2s.coords['time'] = argo.time

for i, (g, argoi) in enumerate(argo.groupby('time')):
    argois = argoi.sortby('sg0')
    argois['z'] = argoi.z
    ptmp = gsw.p_from_z(-argoi.z, lat=lat)
    N2, pmid = gsw.Nsquared(argois.SA, argois.CT, ptmp, lat=lat)
    N2z = -gsw.z_from_p(pmid, lat=lat)
    N2s[:,i] = N2
N2s.coords['z'] = N2z
N2s = N2s.where(np.isfinite(N2s), drop=True)
N2s = N2s.where(N2s>0, np.nan)


N2deep = xr.full_like(N2s, np.nan)
N2deep = N2deep.isel(z=-1)
N2deep['z'] = bottom_depth
N2deep.values = N2s.isel(z=-1)

N2shallow = N2deep.copy()
N2shallow['z'] = 0
N2shallowvalues = N2s.isel(z=0)

N2s = xr.concat([N2shallow, N2s, N2deep], dim='z')
N2s = N2s.transpose('z', 'time')

N2s = N2s.interpolate_na(dim='z')

# %%
fig, ax = gv.plot.quickfig()
np.log10(N2s).plot(hue='time', y='z', add_legend=False, yincrease=False);
ax.set(ylim=(3000, 0))

# %% [markdown]
# ok, looks great. Now interpolate to regular depth grid.

# %%
zmax = bottom_depth + 10 - bottom_depth % 10
print(zmax)

# %%
znew = np.arange(0, zmax, 10)
N2s = N2s.interp(z=znew)

# %%
fig, ax = gv.plot.quickfig()
np.log10(N2s).plot(hue='time', y='z', add_legend=False, yincrease=False);
ax.set(ylim=(3000, 0))

# %%
tmp = N2s.isel(time=11)
modes = gv.ocean.vmodes(tmp.z.data, np.sqrt(tmp.data), lat, 4)

# %%
mm = []
for g, ni in N2s.groupby('time'):
    mm.append(gv.ocean.vmodes(ni.z.data, np.sqrt(ni.data), lat, 4))

# %%
fig, ax = gv.plot.quickfig(w=3, yi=False)
for mi in mm:
    ax.plot(mi[0][:,1], tmp.z, color='0.2', lw=0.5)
    ax.plot(mi[0][:,2], tmp.z, color='0.4', lw=0.5)
    ax.plot(mi[0][:,3], tmp.z, color='0.6', lw=0.5)
ax.set(ylabel='depth [m]', xlabel='mode amplitude (normalized)');
gv.plot.png('modes_seasonal_cycle')

# %% [markdown]
# #### Temperature gradient.

# %%
argo['th'] = (['z', 'time'], gsw.pt_from_CT(argo.SA, argo.CT).data)

# %%
Tz = argo.th.differentiate('z')

# %%
Tz.plot(hue='time', add_legend=False, y='z', yincrease=False);

# %%
fig, ax = gv.plot.quickfig()
Tz.where(Tz>0).plot()
ax.set(ylim=(300, 0))

# %% [markdown]
# Let's use sorted $\theta$ to do this. Actually, why not CT?

# %%
tmp = argo.CT.isel(time=11)

# %%
tmp = tmp.where(np.isfinite(tmp), drop=True)

# %%
plt.plot(tmp.sortby(tmp, ascending=False).data)
plt.plot(tmp.data)

# %%
Tzs = xr.full_like(Tz, np.nan)

# %%
CT = argo.CT.where(np.isfinite(argo.CT), drop=True)

tz = []
for i, (g, CTi) in enumerate(CT.groupby('time')):
    CTs = CTi.sortby(CTi, ascending=False)
    CTs['z'] = CTi.z.data
    tz.append(CTs.differentiate('z'))

Tzs = xr.concat(tz, dim='time')

# %%
Tzs.plot(y='z', yincrease=False)

# %% [markdown]
# ### Argo Climatology Functional Form

# %%
N2, Tz = niskine.clim.climatology_argo_woce(lon, lat, bottom_depth)

# %%
fig, ax = gv.plot.quickfig()
np.log10(N2).plot(yincrease=False)

# %%
fig, ax = gv.plot.quickfig()
Tz.plot(yincrease=False);

# %% [markdown]
# ### Interpolate to time vector

# %% [markdown]
# Interpolate the seasonal climatology to the actual time vector of the mooring, this will make most calculations much easier.

# %%
adcp = niskine.io.load_gridded_adcp(mooring=1)

# %%
time = adcp.time.copy()

# %%
time.dt.month.gv.tplot()

# %% [markdown]
# Let's generate a time vector that has one data point at the 15th of each month. Line up the climatology with this and then interpolate to hourly values. Even easier would be to select the climatological value by month, but would that introduce a certain steppiness in the results?

# %%
start_year = time[0].dt.year.data
end_year = time[-1].dt.year.data
years = end_year - start_year +1
startstr = '{}-01-15'.format(start_year)
endstr = '{}-12-15'.format(end_year)
timenew = pd.date_range(start=startstr, end=endstr, periods=12*years)

# %%
years

# %%
timeax = 0 if N2.shape[0] == 12 else 1
assert N2.dims[timeax] == "time"

# %%
tmp = N2.data
if years > 1:
    tmp2 = np.concatenate([tmp, tmp], axis=1)
else:
    tmp2 = tmp
if years > 2:
    yy = years - 2
    while yy > 0:
        tmp2 = np.concatenate([tmp2, tmp], axis=1)
        yy -= 1

# %%
N3 = xr.DataArray(tmp2,
                  coords={'time': (['time'], timenew), 'z': (['z'], N2.z.data)},
                  dims=['z', 'time'])

# %%
N3 = N3.interp_like(time)

# %%
np.log10(N3).gv.tcoarsen().gv.tplot()

# %% [markdown]
# Now in functional form:

# %%
N3 = niskine.clim.interpolate_seasonal_data(time, N2)

# %%
np.log10(N3).gv.tcoarsen().gv.tplot()
