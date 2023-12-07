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
# %matplotlib inline
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import metpy
from pathlib import Path
import cartopy.crs as ccrs

import gvpy as gv

import niskine

# %reload_ext autoreload
# %autoreload 2

# %config InlineBackend.figure_format = 'retina'

# %%
gv.plot.helvetica()

# %%
cfg = niskine.io.load_config()

# %%
# load mooring locations
locs = xr.open_dataset(cfg.mooring_locations)
m1lon = locs.sel(mooring=1).lon_actual.item()
m1lat = locs.sel(mooring=1).lat_actual.item()

# %% [markdown]
# # NISKINe SSH Analysis

# %% [markdown]
# Calculate EKE as variance of eddy currents following [Heywood et al. 1994](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/94JC01740)

# %% [markdown]
# Here we are looking at SSH data downloaded for the time of the NISKINe moorings but also starting at 2005 to have some climatological statistics.

# %% [markdown]
# I am still not 100% sure whether to use absolute geostrophic velocities or geostrophic velocity anomalies when calculating eddy kinetic energy and vorticity. I am super convinced though that it should be the anomalies, they are referenced to the [1993, 2012] period and thus calculated in the same way as sea level anomalies (`sla`).

# %% [markdown]
# ### Load Data

# %%
alt = niskine.io.load_ssh()

# %%
# we could also load hourly data (but don't have those for the whole time series)
# ssh = niskine.io.load_ssh(hourly=True)

# %% [markdown]
# Resolution in lat/lon:

# %%
print(alt.lat.diff(dim="lat").isel(lat=0).item())
print(alt.lon.diff(dim="lon").isel(lon=0).item())

# %% [markdown]
# Load Smith & Sandwell

# %%
ss = gv.ocean.smith_sandwell(lon=alt.lon, lat=alt.lat)

# %% [markdown]
# ### Calculate EKE

# %%
alt['eke'] = 1/2 * (alt.ugosa**2 + alt.vgosa**2)

# %% [markdown]
# ### Calculate Vorticity

# %%
f = gv.ocean.inertial_frequency(m1lat)

# %% [markdown]
# `metpy` has a routine for vorticity calculation that is aware of lon/lat grids.

# %%
vort = metpy.calc.vorticity(alt.ugosa, alt.vgosa)
# metpy adds units using pint - let's get rid of that for now
vort = vort.metpy.dequantify()

# %%
vort = vort / f

# %%
vort.attrs = dict(long_name='$\zeta / f$', units='')

# %% [markdown]
# As a sanity check, plot sla contours over $\zeta/f$.

# %%
vort.isel(time=10).plot(cbar_kwargs=dict(label="$\zeta/f$", aspect=30, shrink=0.7), vmin=0.3, vmax=-0.3, cmap="RdBu")
(alt.sla.isel(time=10)-alt.sla.mean(dim="time")).plot.contour(colors="k")

# %% [markdown]
# `xgcm` can also do this? In principle yes, but not straight out of the box. Leaving this here but not sure if I want to pursue this any further...

# %%
# import xgcm
# from xgcm import as_grid_ufunc

# grid = xgcm.Grid(alt)

# def diff_forward_1d(a):
#     return a[..., 1:] - a[..., :-1]

# def diff(arr, axis):
#     """First order forward difference along any axis"""
#     return np.apply_along_axis(diff_forward_1d, axis, arr)
 

# @as_grid_ufunc(
#     "(X:center,Y:center),(X:center,Y:center)->(X:center,Y:center)",
#     boundary_width={"X": (1, 0), "Y": (1, 0)},
# )
# def vorticity(u, v):
#     v_diff_x = diff(v, axis=-2)
#     u_diff_y = diff(u, axis=-1)
#     return v_diff_x[..., 1:] - u_diff_y[..., 1:, :]

# vort = vorticity(grid, alt.ugosa, alt.vgosa, axis = [("X", "Y"), ("X", "Y")])

# %% [markdown]
# Interpolate to M1 location.

# %%
m1vort = vort.interp(lon=m1lon, lat=m1lat)
m1eke = alt.eke.interp(lon=m1lon, lat=m1lat)

# %% [markdown]
# Also interpolate to M2 and M3.

# %%
m2vort = vort.interp(lon=locs.sel(mooring=2).lon_actual, lat=locs.sel(mooring=2).lat_actual)
m2eke = alt.eke.interp(lon=locs.sel(mooring=2).lon_actual, lat=locs.sel(mooring=2).lat_actual)

# %%
m3vort = vort.interp(lon=locs.sel(mooring=3).lon_actual, lat=locs.sel(mooring=3).lat_actual)
m3eke = alt.eke.interp(lon=locs.sel(mooring=3).lon_actual, lat=locs.sel(mooring=3).lat_actual)

# %% [markdown]
# Also interpolate to OSNAP MM4.

# %%
mm4lat = 57.992698669433594
mm4lon = -21.14349937438965
mm4vort = vort.interp(lon=mm4lon, lat=mm4lat)
mm4eke = alt.eke.interp(lon=mm4lon, lat=mm4lat)

# %%
ax = m1vort.sel(time=slice("2019-05", "2020-10")).gv.tplot(label="M1")
ax = m2vort.sel(time=slice("2019-05", "2020-10")).gv.tplot(ax=ax, label="M2")
ax = m3vort.sel(time=slice("2019-05", "2020-10")).gv.tplot(ax=ax, label="M3")
ax.legend()

# %% [markdown]
# We see northward low-mode flux at OSNAP MM4 in early 2016. Is there something in the vorticity?

# %%
ax = mm4vort.sel(time=slice("2014-07", "2016-07")).gv.tplot()

# %%
print(mm4vort.sel(time=slice("2015-01", "2015-03")).mean().item())
print(mm4vort.sel(time=slice("2016-01", "2016-03")).mean().item())

# %%
ax = mm4vort.sel(time=slice("2015-01", "2015-03")).gv.tplot()
ax = mm4vort.sel(time=slice("2016-01", "2016-03")).gv.tplot(ax=ax)

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 5),
                       constrained_layout=True)

vort.sel(time=slice("2015-01", "2015-03")).mean(dim="time").plot(ax=ax[0], cbar_kwargs=dict(label="$\zeta/f$", aspect=30, shrink=0.7, orientation="horizontal"), vmin=0.3, vmax=-0.3, cmap="RdBu_r")
ax[0].set(title="Jan-Mar 2015")

vort.sel(time=slice("2016-01", "2016-03")).mean(dim="time").plot(ax=ax[1], cbar_kwargs=dict(label="$\zeta/f$", aspect=30, shrink=0.7, orientation="horizontal"), vmin=0.3, vmax=-0.3, cmap="RdBu_r")
ax[1].set(title="Jan-Mar 2016")


for axi in ax:
    axi.plot(mm4lon, mm4lat, "kx")
    axi.annotate("OSNAP\nMM4", xy=(mm4lon, mm4lat), xytext=(10, -30), textcoords="offset points")
    axi.set(xlabel="", ylabel="")
niskine.io.png("OSNAP_winter_average_vorticity")

# %% [markdown]
# Look at meridional gradients

# %%
fig, ax = gv.plot.quickfig()
vort.sel(time=slice("2015-01", "2015-03")).sel(lon=mm4lon, method="nearest").sel(lat=slice(mm4lat, mm4lat+1)).plot(x="time", ax=ax, cbar_kwargs=dict(label="$\zeta/f$", aspect=30, shrink=0.7, orientation="horizontal"), vmin=0.3, vmax=-0.3, cmap="RdBu_r")

# %%
fig, ax = gv.plot.quickfig()
vort.sel(time=slice("2016-01", "2016-03")).sel(lon=mm4lon, method="nearest").sel(lat=slice(mm4lat, mm4lat+1)).plot(x="time", ax=ax, cbar_kwargs=dict(label="$\zeta/f$", aspect=30, shrink=0.7, orientation="horizontal"), vmin=0.3, vmax=-0.3, cmap="RdBu_r")

# %%
mean_2016 = vort.sel(time=slice("2016-01", "2016-03")).sel(lon=mm4lon, method="nearest").sel(lat=slice(mm4lat, mm4lat+1)).mean(dim="time")

# %%
vdiff = np.diff(mean_2016.data[[0, -1]])

# %%
vdiff

# %%
dist = gsw.distance(lon=np.tile(mean_2016.lon.data, 2), lat=mean_2016.lat[[0, -1]]) * 1e3

# %%
dist

# %%
vort_grad = (vdiff/dist).item()

# %%
vort_grad

# %%
beta = gv.ocean.beta(mm4lat)

# %%
beta

# %%
vort_grad / beta

# %%
fig, ax = gv.plot.quickfig()
m1vort.plot.hist(bins=30, color="0.5", density=True)
m1vort.sel(time=slice("2019-05", "2020-10")).plot.hist(bins=30, density=True, alpha=0.3)
plt.axvline(x=0,color='0.3',linestyle='--')

# %% [markdown]
# Save M1 vorticity and EKE to use elsewhere.

# %%
out = xr.Dataset(data_vars=dict(vort=m1vort, eke=m1eke))

# %%
out.to_netcdf(cfg.data.ssh_m1)

# %% [markdown]
# ### Mean vorticity and mooring locations

# %% [markdown]
# For the whole 2005-2020 altimetry record.

# %%
mean_vort = vort.mean(dim="time")
mean_vort.interp(lon=locs.lon_actual, lat=locs.lat_actual)

# %%
0.00229617 - -0.00371561

# %%
mean_vort.where(mean_vort.lat<62,drop=True).plot()
print(mean_vort.where(mean_vort.lat<62,drop=True).min())
print(mean_vort.where(mean_vort.lat<62,drop=True).max())

# %%
fig, ax = gv.plot.quickfig()
mean_vort.plot(cbar_kwargs=dict(label="$\zeta/f$", aspect=30, shrink=0.7), vmin=0.03, vmax=-0.03, cmap="RdBu_r")
ax.plot(locs.lon_actual, locs.lat_actual, "kx")
ax.set(xlim=(-25, -18), ylim=(56, 62))

# %% [markdown]
# Only the mooring period.

# %%
mean_vort_niskine = vort.sel(time=slice("2019-05", "2020-10")).mean(dim="time")
mean_vort_niskine_at_moorings = mean_vort_niskine.interp(lon=locs.lon_actual, lat=locs.lat_actual)

# %%
mean_vort_niskine_at_moorings

# %%
fig, ax = gv.plot.quickfig()
mean_vort_niskine.plot(cbar_kwargs=dict(label="$\zeta/f$", aspect=30, shrink=0.7), vmin=0.03, vmax=-0.03, cmap="RdBu_r")
ax.plot(locs.lon_actual, locs.lat_actual, "kx")
ax.set(xlim=(-25, -18), ylim=(56, 62))

# %% [markdown]
# Compare vorticity gradients between moorings with $\beta$. Let's focus on the M1 - M2 difference. I think here we want to calculate the lateral vorticity gradient from absolute vorticity, not relative to f.

# %%
dist = gsw.distance(lon=locs.lon_actual[:-1], lat=locs.lat_actual[:-1])

# %%
vdiff = mean_vort_niskinem_at_moorings[:-1].diff(dim="mooring") * f

# %%
vort_grad = (-vdiff/dist).item()

# %%
beta = gv.ocean.beta(locs.lat_actual.mean().data)

# %%
print(vort_grad)
print(beta)
print(vort_grad/beta)

# %% [markdown]
# ### Monthly Mean EKE

# %%
eke_climatology = alt.eke.groupby('time.month').mean('time')


# %%
def gl_format(ax):
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top=False
    gl.xlines=False
    gl.ylabels_right=False
    gl.ylines=False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return gl


# %%
fig = plt.figure(figsize=(20, 5))
h = eke_climatology.plot(col='month', col_wrap=4, cmap='magma', cbar_kwargs={'shrink': 0.75})
for i, ax in enumerate(h.axes.flatten()):
    ss.plot.contour(levels=np.arange(-3000, 1000, 1000), colors='w', linestyles='-', linewidths=0.5, ax=ax)
    ax.set(xlabel='', ylabel='', title='month {}'.format(i+1))
#     gl_format(ax)
h.cbar.set_label('EKE')
# plt.savefig('eke_climatology.png', dpi=200, bbox_inches='tight')

# %% [markdown]
# ### EOF analysis

# %%
from eofs.xarray import Eof

# %%
solver = Eof(alt.eke)

# %%
eof1 = solver.eofsAsCovariance(neofs=4)
pc1 = solver.pcs(npcs=4, pcscaling=1)

# %%
h = eof1.isel(mode=range(4)).plot(col='mode')
alt.eke.mean(dim='time').plot.contour(cmap='Purples', ax=h.axes[0][0])
for i, ax in enumerate(h.axes[0]):
    ss.plot.contour(levels=np.arange(-3000, 500, 500), colors='k', linestyles='-', linewidths=0.5, ax=ax)
    ax.set(ylabel='', xlabel='', title='mode {}'.format(i))
# plt.savefig('eof_first_4_modes.png', dpi=200, bbox_inches='tight')

# %%
pc1.isel(mode=range(4)).plot(col='mode');

# %%
explained = solver.varianceFraction()

# %%
explained[:10].plot(marker='o')

# %%
