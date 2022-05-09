# -*- coding: utf-8 -*-
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

# %% [markdown] heading_collapsed=true
# ### Imports

# %% hidden=true
# %matplotlib inline
from pathlib import Path
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import gsw
import cmocean

import gvpy as gv

import niskine

# %config InlineBackend.figure_format = 'retina'

# %reload_ext autoreload
# %autoreload 2
# %autosave 300


# %% [markdown] hidden=true
# Load configuration from `config.yml` in the root directory. `io.load_config()` automatically detects the root directory and adjusts the paths.

# %% hidden=true
conf = niskine.io.load_config()

# %% [markdown] hidden=true
# Link data files into package data directory. Save to run if the links already exist.

# %% hidden=true
mooringdir = '/Users/gunnar/Projects/niskine/data/NISKINe/Moorings/NISKINE19/'
niskine.io.link_proc_adcp(mooringdir=mooringdir)

# %% [markdown] hidden=true
# Make output directory if it doesn't exist yet.

# %% hidden=true
conf.data.gridded.adcp.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# # Merge NISKINe ADCPs

# %% [markdown] heading_collapsed=true
# ## M1

# %% [markdown] hidden=true
# Full time series, do not include short-lived ADCPs.

# %% hidden=true
ma = niskine.merge.MergeADCP(mooring=1, method='simple')

# %% hidden=true
overlap = niskine.merge.determine_overlap(ma.adcps_sorted_ti_zi)

# %% hidden=true
overlap.sum(dim='adcp').plot()

# %% hidden=true
ma.merged.u.gv.tcoarsen().gv.tplot(vmin=-0.7, vmax=0.7)

# %% [markdown] hidden=true
# Save without filled gaps

# %% hidden=true
niskine.merge.save_merged(ma.merged, suffix='simple_merge')

# %% [markdown] hidden=true
# Fill gaps and save.

# %% hidden=true
ma.fill_gaps()

# %% hidden=true
ma.mergedf.u.gv.tcoarsen().gv.tplot(vmin=-0.7, vmax=0.7)

# %% hidden=true
niskine.merge.save_merged(ma.mergedf, suffix='simple_merge_gaps_filled')

# %% [markdown] hidden=true
# Beginning of time series with all ADCPs.

# %% hidden=true
ma = niskine.merge.MergeADCP(
    mooring=1, method="median", stop="2019-06-01", min_end_time=None, dropna=False
)

# %% hidden=true
ma.merged.u.gv.tplot(vmin=-0.7, vmax=0.7)

# %% hidden=true
niskine.merge.save_merged(ma.merged, suffix='may2019_simple_merge')

# %% [markdown] heading_collapsed=true
# ## M2

# %% [markdown] hidden=true
# Full time series, do not include short-lived ADCPs.

# %% hidden=true
ma = niskine.merge.MergeADCP(mooring=2, method='simple')

# %% hidden=true
ma.merged.u.gv.tcoarsen().gv.tplot(vmin=-0.7, vmax=0.7)

# %% [markdown] hidden=true
# Save without filled gaps

# %% hidden=true
niskine.merge.save_merged(ma.merged, suffix='simple_merge')

# %% [markdown] hidden=true
# Fill gaps and save.

# %% hidden=true
ma.fill_gaps()

# %% hidden=true
ma.mergedf.u.gv.tcoarsen().gv.tplot(vmin=-0.7, vmax=0.7)

# %% hidden=true
niskine.merge.save_merged(ma.mergedf, suffix='simple_merge_gaps_filled')

# %% [markdown] hidden=true
# Beginning of time series with all ADCPs.

# %% hidden=true
ma = niskine.merge.MergeADCP(
    mooring=2, method="median", stop="2019-06-01", min_end_time=None, dropna=False
)

# %% hidden=true
ma.merged.u.gv.tplot(vmin=-0.7, vmax=0.7)

# %% hidden=true
niskine.merge.save_merged(ma.merged, suffix='may2019_simple_merge')

# %% [markdown] heading_collapsed=true
# ## M3

# %% [markdown] hidden=true
# Full time series, do not include short-lived ADCPs.

# %% hidden=true
ma = niskine.merge.MergeADCP(mooring=3, method='simple')

# %% hidden=true
ma.merged.u.gv.tcoarsen().gv.tplot(vmin=-0.7, vmax=0.7)

# %% [markdown] hidden=true
# No need to save just one ADCP...

# %% hidden=true
# niskine.merge.save_merged(ma.merged, suffix='simple_merge')

# %% [markdown] hidden=true
# Beginning of time series with all ADCPs.

# %% hidden=true
ma = niskine.merge.MergeADCP(
    mooring=3, method="median", stop="2019-06-01", min_end_time=None, dropna=False
)

# %% hidden=true
ma.merged.u.gv.tplot(vmin=-0.7, vmax=0.7)

# %% hidden=true
niskine.merge.save_merged(ma.merged, suffix='may2019_simple_merge')
