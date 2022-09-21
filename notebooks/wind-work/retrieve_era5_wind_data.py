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
# #### Imports

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "80aa11a68a82c8", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []} hidden=true
# %matplotlib inline
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
import xarray as xr
import cdsapi

import gvpy as gv
# import osnap
import niskine

# %reload_ext autoreload
# %autoreload 2

# %config InlineBackend.figure_format = 'retina'

# %% hidden=true
conf = niskine.io.load_config()

# %% [markdown]
# # Retrieve ERA5 Wind Data

# %% [markdown]
# Instructions on downloading ERA5 data: https://retostauffer.org/code/Download-ERA5/
#
# Need to install `cdsapi` via conda or pip.
#
# The user key/api is stored in `~/.cdsapirc`.

# %% [markdown]
# Go here to generate your api request: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form

# %% [markdown]
# Data are saved to a `.grib` file that can be read with `xarray`.

# %% [markdown]
# **Update 2022-09-20:** Extending the time series to include the years 2014-2020, thereby also covering the OSNAP time period from 2014 to 2016. Not downloading 2017 and 2018 as the request would be too big. May need to split this up into two requests.

# %% [markdown]
# **Update 2022-09-21:** I am changing this to download one year per request. That way we have the complete wind time series from 2014 to 2020.

# %%
redownload = False


# %%
def download_era5_wind_grib(year):
    gribfile = conf.data.wind.dir.joinpath(f'era5_uv_10m_{year}.grib').as_posix()

    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind',
            ],
            'year': [
                year,
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                60, -27, 57,
                -19,
            ],
        },
        gribfile)


# %%
years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
if redownload:
    for year in years:
        download_era5_wind_grib(year)

# %% [markdown]
# Read grib files

# %%
gribfiles = [conf.data.wind.dir.joinpath(f'era5_uv_10m_{year}.grib').as_posix() for year in years]
wind = [xr.open_dataset(gribfile, engine='cfgrib') for gribfile in gribfiles]

# %% [markdown]
# Concatenate all years into one dataset

# %%
era5uvwind = xr.concat(wind, dim='time')

# %% [markdown]
# Let's make this file a little nicer to work with.

# %%
era5uvwind = era5uvwind.drop(['number', 'step', 'surface', 'valid_time'])
era5uvwind = era5uvwind.rename({'latitude':'lat', 'longitude': 'lon'})

# %%
era5uvwind.time.attrs = dict(long_name=' ')
era5uvwind.u10.attrs = dict(long_name='u$_{\mathrm{10m}}$', units='m/s')
era5uvwind.v10.attrs = dict(long_name='v$_{\mathrm{10m}}$', units='m/s')

# %% [markdown]
# Plot the u-component for one location

# %%
era5uvwind.u10.isel(lon=10, lat=10).plot();

# %%
ncfile = conf.data.wind.dir.joinpath(f'era5_uv_10m.nc').as_posix()

# %%
ncfile

# %%
era5uvwind.to_netcdf(ncfile)

# %% [markdown]
# Read data in netcdf format and have a quick look by contrasting variance in January and June.

# %%
era5uvwind = xr.open_dataset(conf.data.wind.era5)

# %%
era5uvwind.close()

# %%
plot_options = dict(vmin=20, vmax=50)
era5uvwind.u10.sel(time='2020-01').var(dim='time').plot(**plot_options);

# %%
era5uvwind.u10.sel(time='2020-06').var(dim='time').plot(**plot_options);
