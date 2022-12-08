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
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import gsw

import ctdproc
import gvpy as gv

import niskine

# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'

# %% [markdown]
# # NISKINE Shipboard CTD

# %% [markdown]
# Read NISKINe CTD profiles from the 2019 and 2020 cruises and pick full depth profiles near M1. These can then be used to come up with a temperature-density relationship.

# %%
# M1 mooring location
m1lon, m1lat, m1dep = niskine.io.mooring_location(mooring=1)

# %% [markdown]
# ## Cruise 1 (2019)

# %% [markdown]
# For cruise 1 we have profiles processed into .mat format.

# %%
# Read all processed files from cruise 1 (deployment cruise). They are in .mat
# format.
cr1_path = Path("/Users/gunnar/Projects/niskine/cruises/cruise1/data/ctd/proc")
file_list = sorted(cr1_path.glob("*.mat"))

def mat_to_nc(file):
    m = gv.io.loadmat(file)
    ctd = gv.io.mat2dataset(m['datad_1m'])
    return ctd

c = []
for file in file_list:
    c.append(mat_to_nc(file))

for ci in c:
    # doing a weird thing here for longitude - it looks like it is missing the
    # minus sign at times which seems to stem from a parsing error (maybe we
    # should still run this through the ctdproc code...). anyways, making sure
    # that longitude is smaller than zero for all casts.
    ci.attrs["mlon"] = -1 * np.absolute(ci.lon.mean().item())
    ci.attrs["mlat"] = ci.lat.mean().item()

# The cruise report says we did full depth profiles for casts 2, 23, 64, 107.
# The data here show indices 0, 21, 60, 61, 103 have depth > 2000m.


# calculate distance between mooring and ctd casts? maybe just for the deep casts?

# %%
fig, ax = gv.plot.quickfig()
ax.plot(m1lon, m1lat, 'r.')
ax.plot(c[0].mlon, c[0].mlat, 'mo')
ax.plot(c[21].mlon, c[21].mlat, 'go')
for ci in c:
    ax.plot(ci.mlon, ci.mlat, 'k.')

# %% [markdown]
# Okay, looks like the green dot is next to M1. That's cast 23 (index 21). Let's look at this one.

# %%
ctd = c[21]

# %%
fig, ax = gv.plot.quickfig()
ax.plot(ctd.theta1, ctd.depth)
ax.plot(ctd.theta2, ctd.depth)
ax.invert_yaxis()

# %% [markdown]
# ## Cruise 2 (2020)

# %% [markdown]
# I have a couple .cnv files but they are not depth-gridded. I will just run the .hex files through ctdproc and hopefully this will just work™. It looks like cast 10 and 11 are the ones that I want.

# %%
cr2_raw_path = Path("/Users/gunnar/Projects/niskine/cruises/cruise2/data/ctd/raw")

# %%
hex_files = sorted(list(cr2_raw_path.glob("*.hex")))

# %%
hex_files


# %%
def proc_hex(hexfile):
    cx = ctdproc.io.CTDx(hexfile)
    cx_ud = ctdproc.proc.run_all(cx)

    dz = 1
    zmin = 10
    zmax = np.floor(cx_ud['down'].depth.max().data)
    datad1m = ctdproc.proc.bincast(cx_ud['down'], dz, zmin, zmax)
    datau1m = ctdproc.proc.bincast(cx_ud['up'], dz, zmin, zmax)
    return datad1m, datau1m


# %%
cast10_d, cast10_u = proc_hex(hex_files[0])

# %%
cast11_d, cast11_u = proc_hex(hex_files[1])

# %% [markdown]
# Cast 10 is actually quite a few miles to the south. Cast 11 is right at the mooring site though!

# %%
fig, ax = gv.plot.quickfig()
ax.plot(m1lon, m1lat, 'r.')
ax.plot(c[0].mlon, c[0].mlat, 'mo')
ax.plot(c[21].mlon, c[21].mlat, 'go')
for ci in c:
    ax.plot(ci.mlon, ci.mlat, 'k.')
for cast in [cast11_d]:
    ax.plot(cast.lon.mean(), cast.lat.mean(), 'bo')

# %%
cast10_d.to_netcdf("/Users/gunnar/Projects/niskine/cruises/cruise2/data/ctd/proc/ar47_010_dc_1m.nc")

# %%
cast11_d.to_netcdf("/Users/gunnar/Projects/niskine/cruises/cruise2/data/ctd/proc/ar47_011_dc_1m.nc")
