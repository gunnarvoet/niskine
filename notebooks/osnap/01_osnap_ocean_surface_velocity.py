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

# %% [markdown] heading_collapsed=true
# ##### Imports

# %% hidden=true
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
import gsw
import scipy
import gvpy as gv
import niskine

# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'

# %% hidden=true
conf = niskine.io.load_config()

# %% [markdown]
# # OSNAP Surface Velocity

# %% [markdown]
# The wind work calculation needs an estimate for the ocean surface velocity at the OSNAP moorings.
#
# We can't calculate the mixed layer depth from the sparse CTD data on the OSNAP moorings.
#
# We thus have to estimate the ocean surface velocity via
# - The shallowest velocity observation. This may introduce some noise.
# - Ocean velocity averaged over a constant depth range. Previously I have used 300m.
# - Mixed layer depth from climatology. I think I have the Argo MLD climatology somewhere?

# %%
mld = xr.open_dataset(conf.data.ml.mld_argo)

# %% [markdown]
# Load data for one mooring

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "f9337f7b8cb73", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
m = niskine.osnap.Mooring(moorstr='UMM4')

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "f9337f7b8cb73", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
print(m.lon[0], m.lat[0])

# %%
mld.da.interp(lon=m.lon[0], lat=m.lat[0]).plot(yincrease=False)

# %% [markdown]
# Okay, this becomes really shallow in the summer, too shallow actually to have velocity estimates within the mixed layer.

# %%
m.adcp.u.where(m.adcp.z<200, drop=True).gv.tcoarsen().plot(yincrease=False)

# %%
u = m.adcp.u.where(m.adcp.z<200, drop=True).mean(dim='z')
v = m.adcp.v.where(m.adcp.z<200, drop=True).mean(dim='z')
vel = xr.merge([u, v])

# %%
vel.u.plot()
vel.v.plot()

# %% [markdown]
# Look at the difference between averaging over the top 100 vs top 200m.

# %%
m.adcp.u.where(m.adcp.z<200, drop=True).mean(dim='z').plot()
(m.adcp.u.where(m.adcp.z<100, drop=True).mean(dim='z') - m.adcp.u.where(m.adcp.z<200, drop=True).mean(dim='z')).plot()

# %% [markdown]
# We should probably look at the NI component as well.

# %%
