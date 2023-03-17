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

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "80aa11a68a82c8", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
# %matplotlib inline
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import gvpy as gv
import velosearaptor as trex
import niskine

# %reload_ext autoreload
# %autoreload 2
# %autosave 0

# %config InlineBackend.figure_format = 'retina'

# %%
cfg = niskine.io.load_config()

# %% [markdown]
# Link ADCP files into data dir

# %%
mooringdir = Path('/Users/gunnar/Projects/niskine/data/NISKINe/Moorings/NISKINE19')
niskine.io.link_proc_adcp(mooringdir)

# %%

# %% [markdown]
# # Shallow ADCP Winter Day Time Gaps

# %% [markdown]
# The shallow ADCPs 3109 and 3110 show very short range during day time in the winter months. Investigate this a bit further and find out if data drop out due to a bad setting or due to instrument limitation.

# %%
a = niskine.io.load_adcp(mooring=1, sn=3109)

# %%
t1 = slice('2019-07-09', '2019-07-13')
t2 = slice('2020-01-09', '2020-01-13')

# %%
a.sel(time=t1).u.dropna(dim='z', how='all').gv.tplot()

# %%
a.sel(time=t1).amp.dropna(dim='z', how='all').gv.tplot()

# %%
a.sel(time=t2).u.dropna(dim='z', how='all').gv.tplot()

# %%
a.sel(time=t2).amp.dropna(dim='z', how='all').gv.tplot()

# %%
a.sel(time=t1).pg.dropna(dim='z', how='all').gv.tplot()

# %%
a.sel(time=t2).pg.dropna(dim='z', how='all').gv.tplot()

# %% [markdown]
# Look into raw data. In addition to two ADCPs on M1 we have also the uplooker 300kHz on M2.

# %%
rawfile_m1 = list(mooringdir.joinpath("M1/ADCP/raw/SN3109/").glob("*.000"))
rawfile_m12 = list(mooringdir.joinpath("M1/ADCP/raw/SN9408/").glob("*.000"))
rawfile_m2 = list(mooringdir.joinpath("M2/ADCP/raw/SN3110/").glob("*.000"))

# %%
r1 = trex.io.read_raw_rdi(rawfile_m1)
r12 = trex.io.read_raw_rdi(rawfile_m12)
r2 = trex.io.read_raw_rdi(rawfile_m2)

# %%
r1.z

# %%
r1.sel(beam=beam).amp.rolling(time=4).mean().dropna(dim='z', how='all').gv.tplot(yincrease=False)

# %%
r1.sel(beam=beam, time=slice("2020-02", "2020-04")).amp.rolling(time=4).mean().dropna(dim='z', how='all').gv.tplot(yincrease=False)

# %%
r1.sel(time=slice("2020-02", "2020-04")).pressure.rolling(time=4).mean().gv.tplot()

# %%
r1.amp.sel(beam=beam, z=40, method="nearest").coarsen(time=200, boundary="pad").mean().plot()

r12.amp.sel(beam=beam, z=40, method="nearest").coarsen(time=200, boundary="pad").mean().plot()


# %%
def plot_raw_one_beam_m1(time, beam):
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                           constrained_layout=True, sharex=True)
    opts = dict(vmin=60, vmax=230, cmap="Spectral_r")
    r1.sel(time=time, beam=beam).amp.dropna(dim='z', how='all').gv.tplot(ax=ax[0], yincrease=False, **opts)
    r12.sel(time=time, beam=beam).amp.dropna(dim='z', how='all').gv.tplot(ax=ax[1], **opts)
    opts2 = dict(vmin=0, vmax=150, cmap="magma")
    r1.sel(time=time, beam=beam).cor.dropna(dim='z', how='all').gv.tplot(ax=ax[2], yincrease=False, **opts2)
    r12.sel(time=time, beam=beam).cor.dropna(dim='z', how='all').gv.tplot(ax=ax[3], **opts2)
    
    if np.datetime64(time.start)<np.datetime64("2020-01-01"):
        plt.suptitle("summer @ M1")
    else:
        plt.suptitle("winter @ M1")
            


# %%
beam = 1

# %%
plot_raw_one_beam_m1(t1, beam)
niskine.io.png(f"m1_raw_beam{beam}_summer")

# %%
plot_raw_one_beam_m1(t2, beam)
niskine.io.png(f"m1_raw_beam{beam}_winter")


# %%
def plot_raw_one_beam_m2(time, beam):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7.5, 3),
                           constrained_layout=True, sharex=True)
    opts = dict(vmin=60, vmax=230, cmap="Spectral_r")
    r2.sel(time=time, beam=beam).amp.dropna(dim='z', how='all').gv.tplot(ax=ax[0], yincrease=False, **opts)
    opts2 = dict(vmin=0, vmax=150, cmap="magma")
    r2.sel(time=time, beam=beam).cor.dropna(dim='z', how='all').gv.tplot(ax=ax[1], yincrease=False, **opts2)
    
    if np.datetime64(time.start)<np.datetime64("2020-01-01"):
        plt.suptitle("summer @ M2")
    else:
        plt.suptitle("winter @ M2")
            


# %%
plot_raw_one_beam_m2(t1, beam)
niskine.io.png(f"m2_raw_beam{beam}_summer")

# %%
plot_raw_one_beam_m2(t2, beam)
niskine.io.png(f"m2_raw_beam{beam}_winter")

# %%

# %%

# %%

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for i, axi in enumerate(ax):
    r1.sel(time=t1, beam=i+1).amp.dropna(dim='z', how='all').gv.tplot(ax=axi)

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for i, axi in enumerate(ax):
    r12.sel(time=t1, beam=i+1).amp.dropna(dim='z', how='all').gv.tplot(ax=axi)

# %% [markdown]
# Winter @ M1

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for i, axi in enumerate(ax):
    r1.sel(time=t2, beam=i+1).amp.dropna(dim='z', how='all').gv.tplot(ax=axi)

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for i, axi in enumerate(ax):
    r12.sel(time=t2, beam=i+1).amp.dropna(dim='z', how='all').gv.tplot(ax=axi)

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for i, axi in enumerate(ax):
    r2.sel(time=t2, beam=i+1).amp.dropna(dim='z', how='all').gv.tplot(ax=axi)

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for i, axi in enumerate(ax):
    r1.sel(time=t2, beam=i+1).cor.dropna(dim='z', how='all').gv.tplot(ax=axi)

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for i, axi in enumerate(ax):
    r12.sel(time=t2, beam=i+1).cor.dropna(dim='z', how='all').gv.tplot(ax=axi)

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for i, axi in enumerate(ax):
    r2.sel(time=t2, beam=i+1).cor.dropna(dim='z', how='all').gv.tplot(ax=axi)

# %%
