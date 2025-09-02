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

# %% [markdown] heading_collapsed=true
# #### Imports

# %% hidden=true
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

# %% hidden=true
cfg = niskine.io.load_config()

# %% hidden=true
gv.plot.helvetica()

# %% hidden=true
# load mooring locations
locs = xr.open_dataset(cfg.mooring_locations)
m1lon, m1lat, m1depth = niskine.io.mooring_location(mooring=1)

# %% [markdown]
# # NISKINe Stratification

# %% [markdown]
# Load gridded temperature

# %%
tall = xr.open_dataarray(niskine.io.CFG.data.gridded.temperature)

# %% [markdown]
# Subsample - daily averages for now.

# %%
td = tall.resample(time="1D").mean()

# %%
td.gv.tplot()

# %% [markdown]
# Extract stratification from the WOCE Argo climatology

# %%
n2, tz = niskine.clim.climatology_argo_woce(m1lon, m1lat, m1depth)

# %%
np.log10(n2).plot(yincrease=False, vmin=-6.5, vmax=-4)

# %% [markdown]
# I am using `gsw.Nsquared` in the above - let's see how this compares with doing the vertical differencing ourselves. Here we don't sort density before differencing, therefore a few gaps in the winter. Other than that it's looking pretty similar.
#
# $$N^2 = -g/\rho_0 \frac{\Delta \rho}{\Delta z}$$

# %%
woce = gv.ocean.woce_argo_profile(lon=m1lon, lat=m1lat)
p = gsw.p_from_z(-woce.z, woce.lat)
SA = gsw.SA_from_SP(woce.s, p, woce.lon, woce.lat)

N2 = (9.81 / 1025) * gsw.pot_rho_t_exact(SA, woce.t, p, 0).dropna(
    dim="z"
).differentiate(coord="z")

# %%
np.log10(N2.where(N2>0)).plot(y="z", yincrease=False, vmin=-6.5, vmax=-4);

# %% [markdown]
# Interpolate climatology to M1 times.

# %%
SAd = niskine.clim.interpolate_seasonal_data(td.time, SA.transpose()).dropna(dim="z")
SAd = SAd.rename(z="depth")
SAd = SAd.interp(depth=td.depth)

# %%
sd = niskine.clim.interpolate_seasonal_data(td.time, woce.s.transpose()).dropna(dim="z")
sd = sd.rename(z="depth")
sd = sd.interp(depth=td.depth)


# %% [markdown]
# Sort temperature.

# %%
def sort_da_keep_nans(da, axis=0, ascending=True):
    mask = ~np.isnan(da)
    das = da.copy()
    dass = np.sort(da, axis=axis)
    if not ascending:
        dass = dass[::-1]

    for i, ti in enumerate(dass.transpose()):
        das[mask[:, i], i] = ti[~np.isnan(ti)]
    return das


# %%
tds = sort_da_keep_nans(td, axis=0, ascending=False)

# %%
fig, ax = gv.plot.quickfig(w=8)
tds.plot.contourf(ax=ax, yincrease=False, levels=np.arange(3.5, 13.5, 0.5), cmap="Spectral_r", cbar_kwargs=dict(aspect=30, shrink=0.7, ticks=mpl.ticker.MaxNLocator(5), pad=0.01))

# %%
ax = tds.diff(dim="depth").gv.tplot(cmap="Reds_r")

# %%
tdp = gsw.p_from_z(-td.depth, m1lat)

# %%
tdrho = gsw.pot_rho_t_exact(SAd, tds, tdp, 0)

# %%
tdrhos = sort_da_keep_nans(tdrho, axis=0, ascending=True)

# %%
tdn2 = (9.81 / 1025) * tdrhos.differentiate(coord="depth")

# %%
np.log10(tdn2).gv.tplot(vmin=-6.5, vmax=-4, cbar_kwargs=dict(aspect=30, shrink=0.7, ticks=mpl.ticker.MaxNLocator(5), pad=0.01, label="log$_{10}(N^2)$"), )
niskine.io.png("N")

# %%
tdn2.to_netcdf(cfg.data.gridded.N_daily)

# %% [markdown]
# Okay, it seems hard to use the gridded temperature product to calculate stratification. What do we need it for? WKB scaling, but that could also work just with the climatology.
#
# We do have salinity at a few depths (nominally 48m, 170m, 430m, 707m) and have good depth mappings for these as they have pressure, but I am not sure we can find much of a depth mapping with these.
#
# I think for now I will use the WOCE Argo climatology for WKB scaling. For the plot with the surface fluxes we may just show temperature... We can mention that there is so much TS variability that a mapping of temperature to density is not possible. I think I already mention that in the description of the mixed layer calculation.
#
# What maybe we could do are fits to the temperature data to smooth things out a bit? I am not sure this would be worth the effort though. Another idea would be to just pull the climatology in the direction of the observations but that also seems like a lot of work.

# %% [markdown]
# ---

# %% [markdown]
# Here is a look at the variability in climatological salinity in the upper ocean.

# %%
fig, ax = gv.plot.quickfig()
woce.s.plot(hue="time", y="z", ax=ax)
sd.mean(dim="time").plot(yincrease=False, y="depth", color="k", ax=ax)
ax.set(ylim=[1000, 0])

# %%
