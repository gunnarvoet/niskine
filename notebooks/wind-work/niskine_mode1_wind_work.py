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
import matplotlib as mpl
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

# %%
gv.plot.helvetica()
mpl.rcParams["lines.linewidth"] = 1

# %% [markdown]
# # NISKINe Mode 1 Wind Work

# %% [markdown]
# Calculate near-inertial wind work for the NISKINe mooring M1. Let's see what difference the current feedback correction factor makes.

# %%
Nww = niskine.calcs.NIWindWorkNiskine()

# %% [markdown]
# Run the low-mode flux calculation so we have the mode NI velocity projections that we need for the mode 1 wind work.

# %%
m1 = niskine.mooring.NISKINeMooring(add_bottom_adcp=True, add_bottom_zero=True)

# %%
N = niskine.flux.Flux(mooring=m1, bandwidth=1.06, runall=True, climatology="ARGO")

# %% [markdown]
# Extract mode 1 velocities at the surface.

# %%
m1su = N.up.sel(mode=1, z=0)
m1sv = N.vp.sel(mode=1, z=0)
ax = m1su.gv.tplot()

# %%
m1ww = Nww.taux_ni * m1su + Nww.tauy_ni * m1sv

# %%
m1ww.name = "NISKINE M1 mode1 wind-work"

# %%
m1ww.attrs = dict(long_name='mode 1 wind-work', units='W/m$^2$')

# %%
m1ww

# %%
ax = (Nww.wind_work * 1e3).gv.tplot(label="total NI WW", color="C0")
ax = (m1ww * 1e3).gv.tplot(ax=ax, label="mode 1 NI WW", color="C6")
ax.legend()
ax.set(ylabel="mW/m$^2$");
niskine.io.png("niskine_ni_mode1_wind_work", subdir="wind-work")
niskine.io.pdf("niskine_ni_mode1_wind_work", subdir="wind-work")

# %% [markdown]
# Save mode 1 wind work.

# %%
m1ww.to_netcdf(cfg.data.wind_work.niskine_m1_mode1.as_posix())
