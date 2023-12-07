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

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "80aa11a68a82c8", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
from pathlib import Path
import gsw

import niskine

# %reload_ext autoreload
# %autoreload 2
# %autosave 0

# %config InlineBackend.figure_format = 'retina'

# %%
m1lon, m1lat, m1depth = niskine.io.mooring_location(mooring=1)
m2lon, m2lat, m2depth = niskine.io.mooring_location(mooring=2)
m3lon, m3lat, m3depth = niskine.io.mooring_location(mooring=3)

# %%
gsw.distance(lon=np.array([m1lon, m2lon]), lat=np.array([m1lat, m2lat]))

# %%
gsw.distance(lon=np.array([m1lon, m3lon]), lat=np.array([m1lat, m3lat]))

# %%
gsw.distance(lon=np.array([m2lon, m3lon]), lat=np.array([m2lat, m3lat]))
