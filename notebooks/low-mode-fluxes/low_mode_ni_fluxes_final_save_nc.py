# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: uv-niskine
#     language: python
#     name: uv-niskine
# ---

# %% [markdown]
# #### Imports

# %%
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

# %%
cfg = niskine.io.load_config()

# %%
gv.plot.helvetica()
mpl.rcParams["lines.linewidth"] = 1

# %% [markdown]
# # NISKINe low-mode NI fluxes (final) - save netcdf
# This is the final version!! This notebook only saves netcdf files.

# %% [markdown]
# ## Mooring data structures

# %% [markdown]
# For the NISKINe mooring, add zero NI velocity close to the bottom to constrain the mode fits.

# %%
m1 = niskine.mooring.NISKINeMooring(add_bottom_adcp=True, add_bottom_zero=True)

# %%
m1

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "f9337f7b8cb73", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
os4 = niskine.mooring.OSNAPMooring(moorstr='UMM4')

# %% [markdown]
# ## Calculate fluxes

# %%
N = niskine.flux.Flux(mooring=m1, bandwidth=1.06, runall=True, climatology="ARGO")

# %%
N.flux.fx_ni

# %%
O = niskine.flux.Flux(mooring=os4, bandwidth=1.06, runall=True, climatology="ARGO")

# %% [markdown]
# ## Save results to netcdf

# %%
N.flux.to_netcdf(cfg.data.ni_low_mode_flux_niskine_m1)

# %%
O.flux.to_netcdf(cfg.data.ni_low_mode_flux_osnap_umm4)

# %% [markdown]
# ## Calculate average flux magnitudes and seasonal mean

# %%
Nmag, Ndir = niskine.flux.flux_mag_and_dir(N)
print(f"{Nmag.mean().item():1.1f} {Nmag.units}")

# %%
print(f"highest 5% are in the range {np.percentile(Nmag, 95):1.1f} {Nmag.units} - {np.max(Nmag).item():1.1f} {Nmag.units}")

# %%
Omag, Odir = niskine.flux.flux_mag_and_dir(O)
print(f"{Omag.mean().item():1.1f} {Omag.units}")

# %%
print(f"highest 5% are in the range {np.percentile(Omag, 95):1.1f} {Omag.units} - {np.max(Omag).item():1.1f} {Omag.units}")

# %%
np.percentile(Omag, 95)

# %%
Nmag.plot()

# %%
Omag.plot()
