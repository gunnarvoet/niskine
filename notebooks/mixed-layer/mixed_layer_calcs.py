# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python [conda env:niskine]
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

# %% [markdown]
# # NISKINe Mixed Layer Calculations v2

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
# ## MLD Class

# %%
MixedLayer = niskine.calcs.MixedLayerDepth()

# %% [markdown]
# Zoom into a few days for testing. The subset is stored under `subset`.

# %%
timespan = slice("2019-11-05", "2019-11-10")
MixedLayer.pick_subset(timespan)

# %%
ax = MixedLayer.subset.gv.tplot()

# %% [markdown]
# Okay, this is great, we have some vertical variation in temperature and also some knockdown.

# %% [markdown]
# Generate a time series with the shallowest measurement and see where we observe the warmest temperature.

# %%
MixedLayer.determine_knockdown()

# %%
MixedLayer.depth_shallowest.gv.tplot()

# %% [markdown]
# The following finds the highest temperature and the depth of the highest temperature (which may be deeper than the shallowest observation).

# %%
MixedLayer.determine_highest_temp()

# %%
ax = MixedLayer.depth_shallowest.gv.tplot()
(MixedLayer.depth_shallowest - MixedLayer.depth_highest_temp).plot(ax=ax)
ax.invert_yaxis()

# %% [markdown]
# Calculate mixed layer depth based on a temperature criterion. We look for temperatures that are within $\Theta_\mathrm{high} - \Theta_\mathrm{criterion}$.

# %%
Tcrit = 0.5

# %%
MixedLayer.ptemp.sel(time="2020-02").gv.tplot()

# %%
tmp = MixedLayer.ptemp-MixedLayer.highest_temp

# %% [markdown]
# Blank out all data that are not within criterion range.

# %%
tmp2 = tmp.where(tmp+Tcrit>0)

# %%
tmp2.sel(time="2020-02").gv.tplot()

# %% [markdown]
# Create a mask from the data within range. We are just interested in the deepest data point still in range of the temperature criterion.

# %%
tmp3 = ~np.isnan(tmp2)

# %%
# cmap for False and True
cmap = mpl.colors.ListedColormap(['white', 'steelblue'])

# %%
ax = tmp3.sel(time="2020-02").gv.tplot(cmap=cmap, add_colorbar=False)

# %% [markdown]
# Okay, now we have marked all temperatures within criterion range. Now we want to find the deepest depth within range.

# %% [markdown]
# `argmax()` finds the first occurence of `True` but we want the last occurence. The trick here is reversing the order of the array.
#
# For a two-dimensional array, this would look like this:<br>
# `last_true = a.shape[1] - np.argmax(a[:, ::-1], axis=1) - 1`

# %%
tmp3.sel(time="2020-02").argmax(dim='depth').plot()

# %%
data = tmp3.data

# %%
ind = data.shape[0] - np.argmax(data[::-1, :], axis=0) - 1

# %%
fig, ax = gv.plot.quickfig()
ax.plot(ind)

# %%
mld = xr.DataArray(tmp3.depth.isel(depth=ind), coords=dict(time=tmp3.time.data), dims=["time"])

# %%
ax = tmp3.sel(time="2020-02").gv.tplot(cmap=cmap, add_colorbar=False)
mld.sel(time="2020-02").gv.tplot(ax=ax, color="C3", linewidth=0.5)

# %% [markdown]
# Let's make sure we have the indexing right by selecting the last valid temperature in one profile and comparing it with what we get when using the index for the MLD found above (should come out to zero):

# %%
tmp2.isel(time=100).dropna(dim="depth")[-1].item()-tmp2.isel(time=100)[ind[100]].item()

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
# Let's compare with Anna's product to see if we are doing any better.

# %%
old = xr.open_dataarray(cfg.data.ml.mld_old)

# %%
ax = mld_final.gv.tplot()
ax = old.gv.tplot(ax=ax)

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
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4),
                       constrained_layout=True)
MixedLayer.mld.groupby("time.month").mean().plot(color='C7', linestyle="--", yincrease=False, label='NISKINe M1 MLD no qc')
MixedLayer.mld.where(MixedLayer.mask_knockdown & MixedLayer.mask_sst).groupby("time.month").mean().plot(color='C3', linestyle="-", yincrease=False, label='NISKINe M1 MLD')
MixedLayer.mld_c.groupby("time.month").mean().plot(color='C0', yincrease=False, label='NISKINe M1 MLD w/ Argo')
MixedLayer.argo_mld.da_m1.plot(color='C4', yincrease=False, label='Argo MLD Climatology (algorithm)')
MixedLayer.argo_mld.dt_m1.plot(color='C6', yincrease=False, label='Argo MLD Climatology [threshold]')
ax.legend()
ax.set(ylabel='MLD [m]')
gv.plot.axstyle(ax)
niskine.io.png('mld_argo_and_niskine', subdir='mixed-layer')

# %% [markdown]
# Save a dataset that has initial MLD / masks / combined product. What specifically do we want in there? Just the 0.2°C criterion or more?

# %%
out = xr.Dataset(
    data_vars=dict(
        mld=(("time"), MixedLayer.mld.data), 
        mld_c=(("time"), MixedLayer.mld_c.data),
        sst=(("time"), MixedLayer.ssti.data), 
        mask_sst=(("time"), MixedLayer.mask_sst.data), 
        mask_knockdown=(("time"), MixedLayer.mask_knockdown.data), 
        argo_mld=(("time"), MixedLayer.argo_mld_i.data), 
                  ),
    coords=dict(time=(("time"), MixedLayer.mld.time.data)),
)

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

# %%
