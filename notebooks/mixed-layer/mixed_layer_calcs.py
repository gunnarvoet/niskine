# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: python3 (niskine)
#     language: python
#     name: conda-env-niskine-py
# ---

# %% [markdown]
# #### Imports

# %%
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm.notebook import tqdm
import gsw

import gvpy as gv
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
# # NISKINe Mixed Layer Calculations

# %% [markdown]
# Calculate mixed layer depth and mixed layer velocities at M1.
#
# We can't apply a density-based mixed layer criterion out of the box due to:
# - missing density information (only very few microcats on this mooring and we know there is a lot of isopycnal T-S variability)
# - missing near-surface data during mooring knockdown
#
# We need to use a temperature-only criterion. To get reliable estimates, we sort temperature to increase with depth. Doing this, we loose information on any possible overturns, however, those would result in neutral stratification after a few buoyancy periods anyways so this should not bias the MLD estimate. Sorting temperature also assumes that salinity tends to balance any instabilities in temperature (is that true?).
#
# What we should do in the following
# - Calculate MLD based on sorted temperature.
# - Create a concurrent knockdown time series. This also indicates the first depth bin where we have data on hand.
# - Create a concurrent time series with an SST criterion.
#
# The knockdown and SST threshold time series can be used to blank out unreasonable MLD values. We will have a gappy MLD time series, but one that we trust. We can then either interpolate over the gaps or just work with the gappy data. Or we can fill in with the Argo MLD climatology where needed.
#
# Open questions:
# - convert to potential temperature (and if so before or after sorting?)?
#
# Actually, maybe we should not sort but just use unsorted temperature. Assuming that any temperature inversions are density-balanced, we actually get the best estimate by finding the **deepest** temperature that is still within the criterion.

# %% [markdown]
# Set up two masks:
# - SST criterion
# - knockdown criterion

# %% [markdown]
# We use solely the climatology in July/August when the mixed layer appears to be too shallow for us to measure it.

# %% [markdown]
# All steps are now included in the `MixedLayerDepth` class.

# %%
MixedLayer = niskine.calcs.MixedLayerDepth()

# %%
mld_final = MixedLayer.calc_final_mld_product(tcrit=0.2, sst_criterion=1.5, knockdown_criterion=100)

# %% [markdown]
# Let's compare with the old MLD product to see if we are doing any better.

# %%
old = xr.open_dataarray(cfg.data.ml.mld_old)

# %%
ax = old.gv.tplot(color="0.6", label="previous MLD estimate")
ax = mld_final.gv.tplot(ax=ax, label="new MLD estimate")
ax.invert_yaxis()
ax.legend()
niskine.io.png("mld_compare_with_old_estimate", subdir="mixed-layer")

# %% [markdown]
# Figure with several panels showing how I arrive at the MLD time series.
# - knockdown
# - sst & difference from sst
# - argo mld climatology
# - initial mld time series
# - final mld product with masks applied and filled with argo mld

# %%
MixedLayer.plot_steps_and_final_mld()
niskine.io.png("mld_merged_with_argo_climatology", subdir="mixed-layer")
niskine.io.pdf("mld_merged_with_argo_climatology", subdir="mixed-layer")

# %% [markdown]
# Compare monthly averages with the Argo climatology.

# %%
MixedLayer.plot_argo_climatology_comparison()
niskine.io.png('mld_argo_and_niskine', subdir='mixed-layer')

# %% [markdown]
# Save a dataset that has initial MLD / masks / combined product. What specifically do we want in there? Just the 0.2°C criterion or more?

# %% [markdown]
# Also save only the final MLD product.

# %%
MixedLayer.save_results()

# %% [markdown]
# ## Mixed Layer Velocities

# %% [markdown]
# The following calculates mixed layer velocities and also saves them.

# %%
mlvel = niskine.calcs.mixed_layer_vels()

# %%
fig, ax = gv.plot.quickfig(grid=True, w=8, h=4)
mlvel.u.plot(yincrease=False, color='C4', linewidth=0.8, label='zonal')
mlvel.v.plot(yincrease=False, color='C0', linewidth=0.8, label='meridional')
ax.set(ylabel='vel$_{ML}$ [m/s]', xlabel='', title='NISKINe M1 zonal mixed layer velocity')
gv.plot.concise_date(ax)
ax.legend()
niskine.io.pdf('mlvel', subdir="mixed-layer")
niskine.io.png('mlvel', subdir="mixed-layer")
