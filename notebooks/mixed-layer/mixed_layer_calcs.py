# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
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
conf = niskine.io.load_config()

# %% [markdown]
# # NISKINe Mixed Layer Calculations

# %% [markdown]
# Run a few calculations for the mixed layer. At the moment this includes:
# - read Anna's mixed layer depth estimate and save it in netcdf format
# - calculate mixed layer depth velocities at M1 and save them in netcdf format

# %% [markdown]
# ## Mixed Layer Depth

# %% [markdown]
# Load mixed layer depth as calculated by Anna. Interpolate over gaps and save to netcdf file (name specified in `config.yaml`.

# %%
mld = niskine.io.mld_to_nc()

# %% [markdown]
# Plot mixed layer depth

# %%
fig, ax = gv.plot.quickfig(w=8)
mld.plot(yincrease=False, color='0.1', linewidth=0.8)
ax.set(ylabel='MLD [m]', xlabel='', title='NISKINe M1 Mixed Layer Depth')
niskine.io.png('mld')

# %% [markdown]
# ## Mixed Layer Velocities

# %% [markdown]
# Calculate velocity within the mixed layer (or load if we have run this calculation already, it takes a little bit to loop over all time steps).

# %%
mlvel = niskine.calcs.mixed_layer_vels()

# %%
conf.data.ml.ml_vel

# %%
fig, ax = gv.plot.quickfig(w=8, h=4)
mlvel.u.plot(yincrease=False, color='C4', linewidth=0.8, label='zonal')
mlvel.v.plot(yincrease=False, color='C0', linewidth=0.8, label='meridional')
ax.set(ylabel='vel$_{ML}$ [m/s]', xlabel='', title='NISKINe M1 zonal mixed layer velocity')
gv.plot.concise_date(ax)
ax.legend()
niskine.io.png('mlvel')
