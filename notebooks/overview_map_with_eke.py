# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: python3 (niskine)
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
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.mpl.gridliner import (
    LONGITUDE_FORMATTER,
    LATITUDE_FORMATTER,
)

import gvpy as gv

import niskine

# %reload_ext autoreload
# %autoreload 2

# %config InlineBackend.figure_format = 'retina'

# %%
cfg = niskine.io.load_config()

# %%
ssh = niskine.io.load_ssh(hourly=True)

# %%
gv.plot.helvetica()

# %% [markdown]
# # Overview Map

# %% [markdown]
# Calculate EKE as variance of eddy currents following [Heywood et al. 1994](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/94JC01740)
#
# This notebook was initially used for mooring planning purposes. An old copy is still somewhere in Gunnar's old `niskine/py` directory. Here we are looking at SSH data downloaded for the time of the NISKINe moorings but also starting at 2005 to have some climatological statistics.

# %% [markdown]
# ## Load data

# %%
locs = xr.open_dataset(cfg.mooring_locations).load()

# %%
for g, li in locs.groupby("mooring"):
    print(g)

# %% [markdown]
# ## Altimetry data

# %%
alt = niskine.io.load_ssh()
alt = alt.where((alt.lat<62) & (alt.lat>56), drop=True)

# %%
alt_var = alt.sla.var(dim='time')
alt_var.plot()

# %% [markdown]
# ### Bathymetry

# %% [markdown]
# Load Smith & Sandwell

# %%
ss = gv.ocean.smith_sandwell(lon=alt.lon, lat=alt.lat, pad=(0.2, 0.2), subsample=5).load()

# %%
ss.plot()

# %% [markdown]
# ### EM Float Locations

# %%
cols = [
    "drop number",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "latitude",
    "longitude",
    "dummy",
]
emfloat = pd.read_csv(
    cfg.em_float_locations,
    sep=" ",
    skiprows=[0],
    header=None,
    names=cols,
    index_col=False,
)

# %%
emfloat

# %%
em_time = pd.to_datetime(
    emfloat[["year", "month", "day", "hour", "minute", "second"]]
).to_xarray()

# %%
em_lon = emfloat["longitude"]
em_lat = emfloat["latitude"]

# %%
fig, ax = gv.plot.quickfig()
em_time.plot(ax=ax)

# %%
fig, ax = gv.plot.quickfig()
ax.plot(em_time, em_lat, marker="o", markersize=3)
gv.plot.concise_date(ax)

# %%
fig, ax = gv.plot.quickfig()
ax.plot(em_lon, em_lat)

# %% [markdown]
# ## Plot

# %%
alt.ugos.isel(time=0).plot()
alt.sla.isel(time=0).plot.contour()

# %% [markdown]
# Calculate EKE

# %%
alt['eke'] = 1/2 * (alt.ugosa**2 + alt.vgosa**2)

# %%
alt.eke.isel(time=0).plot()
alt.sla.isel(time=0).plot.contour()
ss.plot.contour(colors='k')

# %%
h = alt.eke.mean(dim="time").plot(cmap="magma")
h.colorbar.set_label("EKE")
ss.plot.contour(
    levels=np.arange(-3000, 500, 500), colors="w", linestyles="-", linewidths=0.5
);

# %%
eke_mean = alt.eke.mean(dim='time')
eke_var = alt.eke.var(dim='time')

# %%
eke_at_m1 = alt.eke.interp(
    lon=locs.sel(mooring=1).lon_actual, lat=locs.sel(mooring=1).lat_actual
)

# %%
eke_at_m1.groupby('time.month').mean('time').plot()

# %%
eke_at_m1.sel(time=slice("2019", "2020")).plot()

# %%
h=eke_var.plot.contourf(levels=15)

# %% [markdown]
# EKE climatology

# %%
eke_climatology = alt.eke.groupby('time.month').mean('time')


# %%
def gl_format(ax):
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.xlines = False
    gl.ylabels_right = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return gl


# %%
fig = plt.figure(figsize=(20, 5))
h = eke_climatology.plot(
    col="month", col_wrap=4, cmap="magma", cbar_kwargs={"shrink": 0.75}
)
for i, ax in enumerate(h.axs.flatten()):
    ss.plot.contour(
        levels=np.arange(-3000, 1000, 1000),
        colors="w",
        linestyles="-",
        linewidths=0.5,
        ax=ax,
    )
    ax.set(xlabel="", ylabel="", title="month {}".format(i + 1))
#     gl_format(ax)
h.cbar.set_label("EKE")
# plt.savefig('eke_climatology.png', dpi=200, bbox_inches='tight')

# %% [markdown]
# EOF analysis

# %%
from eofs.xarray import Eof

# %%
solver = Eof(alt.eke)

# %%
eof1 = solver.eofsAsCovariance(neofs=4)
pc1 = solver.pcs(npcs=4, pcscaling=1)

# %%
h = eof1.isel(mode=range(4)).plot(col="mode")
alt.eke.mean(dim="time").plot.contour(cmap="Purples", ax=h.axs[0][0])
for i, ax in enumerate(h.axs[0]):
    ss.plot.contour(
        levels=np.arange(-3000, 500, 500),
        colors="k",
        linestyles="-",
        linewidths=0.5,
        ax=ax,
    )
    ax.set(ylabel="", xlabel="", title="mode {}".format(i))
# plt.savefig('eof_first_4_modes.png', dpi=200, bbox_inches='tight')

# %%
mode0_corr_annual_cycle = pc1.isel(mode=0).groupby("time.month").mean("time")

# %%
mode0_corr_annual_cycle.plot()

# %%
pc1.isel(mode=range(4)).plot(col="mode");

# %%
pc1.isel(mode=range(4)).groupby("time.month").mean("time").plot(col="mode");

# %%
explained = solver.varianceFraction()

# %%
explained[0]/explained[:5].sum()

# %%
explained[:3].sum()/explained[:5].sum()

# %%
explained[:10].plot(marker="o")

# %%
hsa = gv.maps.HillShade(-ss.data, ss.lon, ss.lat, smoothtopo=3)

# %%
fig, ax = hsa.plot_topo_c(
    cmap="Greys",
)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.5,
    linestyle="-",
)
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.MaxNLocator(nbins=5)
gl.ylocator = mticker.MaxNLocator(nbins=5)
# h = ax.contour(
#     ss.lon,
#     ss.lat,
#     ss.data,
#     levels=np.arange(-3000, 500, 500),
#     colors="0.2",
#     linestyles="-",
#     linewidths=0.75,
#     extend="both",
#     zorder=9,
#     transform=ccrs.PlateCarree(),
# )
ax.set_extent(
    (
        eke_mean.lon.min().data,
        eke_mean.lon.max().data,
        eke_mean.lat.min().data,
        eke_mean.lat.max().data,
    )
)
hsl = ax.contourf(
    eke_mean.lon,
    eke_mean.lat,
    eke_mean.where(eke_mean > 0.006, drop=True).data,
    levels=np.arange(0.008, 0.044, 0.002),
    cmap="magma_r",
    zorder=10,
    alpha=0.5,
    transform=ccrs.PlateCarree(),
    antialiased=True,
)
# show either correlation pattern of 1st eof mode or variance
if 1:
    h = (
        eof1.isel(mode=range(1))
        .squeeze()
        .plot.contour(
            levels=np.arange(0.002, 0.018, 0.002),
            #         levels=np.arange(0.005, 0.025, 0.005),
            colors="0.75",
            alpha=0.8,
            linewidths=1,
            linestyles="--",
            transform=ccrs.PlateCarree(),
            zorder=12,
        )
    )
else:
    h = eke_var.plot.contour(
        levels=8,
        colors="0.75",
        linestyles="--",
        linewidths=1,
        transform=ccrs.PlateCarree(),
        zorder=11,
    )

scale_bar_length = 50
scale_bar_color = "w"
gv.maps.cartopy_scale_bar(
    ax,
    location=(0.05, 0.05),
    length=scale_bar_length,
    plot_kwargs=dict(zorder=20, color=scale_bar_color),
    text_kwargs=dict(zorder=20, fontsize=10, color=scale_bar_color),
)
hcb = plt.colorbar(
    hsl,
    shrink=0.3,
    aspect=30,
    label="mean surface EKE [m$^2$/s$^2$]",
    ticks=[0.01, 0.02, 0.03, 0.04],
)
for g, li in locs.groupby("mooring"):
    ax.plot(
        li.lon_actual,
        li.lat_actual,
        "wo",
        transform=ccrs.PlateCarree(),
        zorder=12,
        markersize=5,
    )
# add location of OSNAP MM4
mm4lat = 57.992698669433594
mm4lon = -21.14349937438965
ax.plot(
    mm4lon,
    mm4lat,
    "ko",
    transform=ccrs.PlateCarree(),
    zorder=12,
    markersize=5,
)
ax.annotate(
    "OSNAP\nMM4",
    xy=(mm4lon, mm4lat),
    xytext=(0, -7),
    textcoords="offset points",
    ha="center",
    va="top",
    color="k",
    fontweight="bold",
    transform=ccrs.PlateCarree(),
    zorder=12,
)

# trajectory of EM-APEX float
ax.plot(
    em_lon,
    em_lat,
    color="w",
    linewidth=0.75,
    marker="o",
    markersize=1,
    transform=ccrs.PlateCarree(),
    zorder=13,
)
for i in [0, 180, 303]:
    ax.plot(
        em_lon[i],
        em_lat[i],
        color="k",
        marker="o",
        markersize=3,
        transform=ccrs.PlateCarree(),
        zorder=14,
    )
    ax.annotate(
        gv.time.datetime64_to_str(em_time[i].data),
        xy=(em_lon[i], em_lat[i]),
        xytext=(4+i/30, 7),
        textcoords="offset points",
        transform=ccrs.PlateCarree(),
        ha="right",
        va="center",
        fontsize=8,
        color="w",
        zorder=14,
    )

m1lon = locs.sel(mooring=1).lon_actual
m1lat = locs.sel(mooring=1).lat_actual
nislon = locs.sel(mooring=2).lon_actual
nislat = locs.sel(mooring=2).lat_actual
ax.annotate(
    "NISKINE",
    xy=(nislon, nislat),
    xytext=(3, -7),
    textcoords="offset points",
    ha="left",
    va="center",
    color="w",
    fontweight="bold",
    transform=ccrs.PlateCarree(),
    zorder=12,
)
ax.annotate(
    "M1",
    xy=(m1lon, m1lat),
    xytext=(13, 12),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color="w"),
    ha="left",
    va="center",
    color="w",
    fontweight="bold",
    transform=ccrs.PlateCarree(),
    zorder=12,
    bbox=dict(pad=0, facecolor="none", edgecolor="none"),
)
# ax.plot(m1lon, m1lat,
#     transform=ccrs.PlateCarree(),
#     color="darkorange",
#     marker="o",
#     markersize="5",
#     zorder=13,
# )
# ax.plot(m1lon, m1lat,
#     transform=ccrs.PlateCarree(),
#     color="k",
#     marker="x",
#     markersize="5",
#     zorder=13,
# )


ax.annotate(
    "EM-APEX Float",
    xy=(-28.3, 57.7),
    transform=ccrs.PlateCarree(),
    ha="center",
    va="center",
    color="w",
    zorder=12,
)

ax.annotate(
    "Hatton Bank",
    xy=(-18, 58),
    transform=ccrs.PlateCarree(),
    ha="center",
    va="center",
    rotation=45,
)

ax.annotate(
    "Reykjanes Ridge",
    xy=(-27, 61),
    transform=ccrs.PlateCarree(),
    ha="center",
    va="center",
    rotation=48,
)

ax.set(title="")
niskine.io.png("map_eke_mean")
niskine.io.pdf("map_eke_mean")

# %%
