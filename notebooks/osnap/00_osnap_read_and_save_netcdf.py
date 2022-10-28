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
import matplotlib as mpl
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

plt.ion()

# %config InlineBackend.figure_format = 'retina'

# %% hidden=true
conf = niskine.io.load_config()

# %% [markdown]
# # OSNAP Mooring data

# %% [markdown]
# ## Read OSNAP Data

# %% [markdown]
# Load data using the `osnap` module of the `niskine` package.
#
# The `Mooring` instance provides interpolated data in the attributes `adcp`, `cm` and `ctd`.

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "f9337f7b8cb73", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
os4 = niskine.osnap.Mooring(moorstr='UMM4')
os4.save_to_netcdf()

# %%
os3 = niskine.osnap.Mooring(moorstr='UMM3')
os3.save_to_netcdf()

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 3),
                       constrained_layout=True, sharey=True)
niskine.osnap.plot_mooring_setup(os3, ax=ax[0])
niskine.osnap.plot_mooring_setup(os4, ax=ax[1])
ax[0].invert_yaxis()

# %%
t1 = '2015-02'
t1 = slice('2015-02-01', '2015-02-17')

# %%
fig, ax = gv.plot.quickfig(w=8)
os4.adcp.u.sel(time=t1).where(np.isfinite(os4.adcp.u), drop=True).plot(yincrease=False)
gv.plot.concise_date()
# gv.plot.png('mm4_adcp_u_2015-02')

# %%
fig, ax = gv.plot.quickfig(w=8)
os4.ctd.sel(time=t1).isel(nomz=range(2)).p.plot(hue='nomz', yincrease=False);
ax.set(title='CTD pressure')
gv.plot.concise_date()
# gv.plot.png('mm4_pressure_shallow_2015-02')

# %%
fig, ax = gv.plot.quickfig(w=8)
os4.ctd.sel(time=t1).isel(nomz=[6]).p.plot(hue='nomz', yincrease=False);
ax.set(title='CTD pressure')
gv.plot.concise_date()
# gv.plot.png('mm4_pressure_deep_2015-02')
