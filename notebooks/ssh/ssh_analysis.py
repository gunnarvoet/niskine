# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# #### Imports

# %%
# # %load /Users/gunnar/Projects/python/standard_imports.py
# %matplotlib inline
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from pathlib import Path
import cartopy.crs as ccrs

import gvpy as gv

import niskine

# %reload_ext autoreload
# %autoreload 2

# %config InlineBackend.figure_format = 'retina'

# %%
conf = niskine.io.load_config()

# %%
ssh = niskine.io.load_ssh(hourly=True)

# %% [markdown]
# # NISKINe SSH Analysis

# %% [markdown]
# Calculate EKE as variance of eddy currents following [Heywood et al. 1994](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/94JC01740)

# %% [markdown]
# This notebook was initially used for mooring planning purposes. An old copy is still somewhere in Gunnar's old `niskine/py` directory. Here we are looking at SSH data downloaded for the time of the NISKINe moorings but also starting at 2005 to have some climatological statistics.

# %% [markdown]
# ## Load data

# %% [markdown]
# ## Altimetry data

# %%
alt = niskine.io.load_ssh()

# %%
alt.sla.var(dim='time').plot()

# %% [markdown]
# ### Bathymetry

# %% [markdown]
# Load Smith & Sandwell

# %%
ss = gv.ocean.smith_sandwell(lon=alt.lon, lat=alt.lat)

# %%
ss.plot()

# %% [markdown]
# Load processed multibeam data

# %%
mb = xr.open_dataarray(conf.data.proc.mb)

# %%
mb

# %% [markdown]
# Reduce in size for faster plotting - don't need the great resolution here.

# %%
mbc = mb.coarsen({'lon': 10, 'lat': 10}, boundary='pad')
b = mbc.mean()

# %%
b.plot()

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
h = alt.eke.mean(dim='time').plot(cmap='magma')
h.colorbar.set_label('EKE')
ss.plot.contour(levels=np.arange(-3000, 500, 500), colors='w', linestyles='-', linewidths=0.5);

# %% [markdown]
# EKE climatology

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
# EOF analysis

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
