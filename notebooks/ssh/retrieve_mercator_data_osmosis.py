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
import niskine

# %reload_ext autoreload
# %autoreload 2
# %autosave 0

# %config InlineBackend.figure_format = 'retina'

# %%
conf = niskine.io.load_config()

# %% [markdown]
# ## Retrieve Mercator Altimetry Data for Osmosis Region

# %% [markdown]
# ### Notes

# %% [markdown]
# Download additional SSH data for Anna.
#
# ` Can I get an EKE estimate for 16°W to 9°W, 44°N to 52°N from September 2012 through September 2013 please?`

# %% [markdown]
# ---

# %% [markdown]
# Here is an example for the options_dict that can be passed to a `RetrieveMercatorData` object. Data are saved to the `data.ssh` directory defined in `config.yml` at the project root level.

# %%
options_dict = {
    "date_min": "2012-09-01 12:00:00",
    "date_max": "2013-10-01 12:00:00",
    "longitude_min": -16.,
    "longitude_max": -9.,
    "latitude_min": 44.,
    "latitude_max": 52.,
    "out_name": "mercator_ssh_osmosis.nc"
}

# %% [markdown]
# ### Download data (daily)

# %% [markdown]
# Download SSH data for 2019 and 2020.

# %%
RM = niskine.io.RetrieveMercatorData(dataset='ssh')

# %% [markdown]
# Note: The following runs for a while without showing any progress while just doing its job. Patience!

# %%
RM.retrieve_data(options_dict)

# %% [markdown]
# ## Quick check

# %%
osmosis_ssh_nc = conf.data.ssh.joinpath(options_dict['out_name'])

# %%
ssh = xr.open_dataset(osmosis_ssh_nc)

# %%
ssh

# %%
ssh.sla.isel(time=0).plot()

# %%
ssh.sla.isel(time=-1).plot()

# %%
