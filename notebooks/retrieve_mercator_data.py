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
# #### Imports

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "80aa11a68a82c8", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []} hidden=true
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
import niskine

# %reload_ext autoreload
# %autoreload 2
# %autosave 0

# %config InlineBackend.figure_format = 'retina'

# %% [markdown]
# ## Retrieve Mercator Altimetry Data

# %% [markdown]
# ### Notes

# %% [markdown]
# Some links on how to access Copernicus data in Python either [via script](https://help.marine.copernicus.eu/en/articles/4899195-how-to-generate-and-run-a-script-to-download-a-subset-of-a-dataset-from-the-copernicus-marine-data-store) or [via python functions](https://help.marine.copernicus.eu/en/articles/5211063-how-to-use-the-motuclient-within-python-environment-e-g-spyder).
#
# This dataset has hourly SSH and derived velocities and also surface temperatures, I think it is a data assimilation product (Mercator model):
# [Global Ocean 1/12° Physics Analysis Model Data](https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/INFORMATION). However, it looks like we can't get data earlier than 2020-01-01 for this. We will download this dataset but let's also try something else in addition!
#
# The [global ocean SSH and derived variables dataset](https://resources.marine.copernicus.eu/product-detail/SEALEVEL_GLO_PHY_L4_MY_008_047/INFORMATION) from Copernicus has SSH and derived velocities from satellite altimetry and is available for the whole mooring deployment period.
#
# I added both data sources to `niskine.io.RetrieveMercatorData`, they can be selected by setting the `dataset` option to either `hourly` or `ssh`.

# %% [markdown]
# ---

# %% [markdown]
# Here is an example for the options_dict that can be passed to a `RetrieveMercatorData` object. Data are saved to the `data.ssh` directory defined in `config.yml` at the project root level.

# %%
options_dict_example = {
    "date_min": "2021-02-21 12:00:00",
    "date_max": "2021-02-27 12:00:00",
    "longitude_min": 100.,
    "longitude_max": 155.,
    "latitude_min": 10.,
    "latitude_max": 40.,
    "out_dir": ".",
}

# %% [markdown]
# ### Download data

# %% [markdown]
# Download SSH data for 2019 and 2020.

# %%
RM = niskine.io.RetrieveMercatorData(dataset='ssh')

# %%
options_dict = dict(out_name='mercator_ssh.nc',
    date_min="2019-01-01 00:00:00",
    date_max="2021-01-01 00:00:00",
    )

# %%
RM.retrieve_data(options_dict)

# %% [markdown]
# Also download the hourly analysis data for 2020. Download file size cannot be over 1024 Mb. Need to download these month by month. Note: This takes a while to download.

# %%
RM = niskine.io.RetrieveMercatorData(dataset='hourly')

# %%
mm = np.arange(np.datetime64("2020-01-01 00:30:00"), np.datetime64("2020-12-31 23:30:00"), dtype='datetime64[M]')

for i in range(10):
    tmin = mm[i] + np.timedelta64(30, 'm')
    tmax = mm[i+1] - np.timedelta64(30, 'm')
    options_dict = dict(out_name=f'hourly_ssh_2020_{i+1:02d}.nc',
        date_min=tmin.astype(str).replace('T', ' ')+':00',
        date_max=tmax.astype(str).replace('T', ' ')+':00',
        )
    print(options_dict['date_max'])
    print(options_dict['out_name'])
    RM.retrieve_data(options_dict)
