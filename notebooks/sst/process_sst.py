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
import niskine

# %reload_ext autoreload
# %autoreload 2

# %config InlineBackend.figure_format = 'retina'

# %%
gv.plot.helvetica()

# %%
cfg = niskine.io.load_config()

# %%
lon, lat, depth = niskine.io.mooring_location(mooring=1)

# %% [markdown]
# # NISKINe SST

# %% [markdown]
# SST data (global fields) were downloaded here. I copied them into the SST data directory for a moment but will move them to the server so they don't clutter my laptop. I am saving the time series for the area of the Iceland Basin.

# %%
# sst_files = sorted(cfg.path.data.joinpath("sst").glob("*.nc"))
# full SST files are also on kipapa at /Volumes/Ahua/data_archive/Climatologies/
sst_files = sorted(Path("/Users/gunnar/Data/sst/").glob("**/20*.nc"))

# %%
sst = xr.open_dataset(sst_files[0])

# %%
masklon = (sst.lon > -25) & (sst.lon < -17)
masklat = (sst.lat > 56) & (sst.lat < 63)

# %%
ssts = sst.where(masklon & masklat, drop=True)
ssts = ssts.drop(["time_bnds", "lat_bnds", "lon_bnds", "sea_ice_fraction", "mask"])

# %%
fig, ax = gv.plot.quickfig()
ssts.analysed_sst.plot()
gv.plot.png("test")

# %%
all_sst = []
for file in sst_files:
    sst = xr.open_dataset(file)
    sst.close()
    ssts = sst.where(masklon & masklat, drop=True)
    ssts = ssts.drop(["time_bnds", "lat_bnds", "lon_bnds", "sea_ice_fraction", "mask"])
    all_sst.append(ssts)

# %%
s = xr.concat(all_sst, dim="time")

# %%
savename = cfg.path.data.joinpath("sst/sst.nc")

# %%
s.to_netcdf(savename)

# %%
s

# %% [markdown]
# Advective scale of a 0.3m/s current:

# %%
0.3*24*3600/1000

# %% [markdown]
# Interpolate SST to mooring M1 location.

# %%
zeroCinK = 273.15

# %%
ssti = s.analysed_sst.interp(lon=lon, lat=lat) - zeroCinK

# %%
ax = ssti.gv.tplot()

# %% [markdown]
# In addition to interpolating SST to the mooring location we probably want to also find the min and max within a radius of about 50km of the mooring.

# %%
50/111.2

# %%
dkm = 20
dlat = dkm/1.852/60
dlon = dlat * np.cos(np.deg2rad(lat))

# %%
masklon = (s.lon > lon-dlon) & (s.lon < lon+dlon)
masklat = (s.lat > lat-dlat) & (s.lat < lat+dlat)

# %%
sa = s.analysed_sst.where(masklon & masklat, drop=True) - zeroCinK

# %%
sa.isel(time=100).plot()

# %%
ax = ssti.gv.tplot()
sa.max(dim=["lon", "lat"]).plot()
sa.min(dim=["lon", "lat"]).plot()

# %%
(sa.max(dim=["lon", "lat"]) - sa.min(dim=["lon", "lat"])).gv.tplot()

# %% [markdown]
# Let's save interpolated / min / max as another dataset for easy access.

# %%
out = xr.merge([dict(sst=ssti, sstmax=sa.max(dim=(["lon", "lat"])), sstmin=sa.min(dim=(["lon", "lat"])))])

# %%
savename = cfg.path.data.joinpath("sst/sst_m1.nc")
out.to_netcdf(savename)

# %%
ax = out.sst.gv.tplot(zorder=5)
ax.fill_between(out.time.data, out.sstmin.data, out.sstmax.data, color="0.8", zorder=4)
ax.set(ylabel="SST [°C]", title="Satellite SST @ NISKINe M1")
niskine.io.png("sst_at_m1", subdir="sst")

# %%
