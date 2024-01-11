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

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "80aa11a68a82c8", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
# %matplotlib inline
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import gsw

import gvpy as gv
import niskine

# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'

# %%
cfg = niskine.io.load_config()

# %%
mld = xr.open_dataarray(cfg.data.ml.mld)

# %% [markdown]
# # NI Filtering

# %% [markdown]
# Apply bandpass filter to ADCP velocity. Mooring knockdown causes some depth levels to have NaN's so we have to deal with those.

# %%
adcp = niskine.io.load_gridded_adcp(mooring=1)
adcp2 = niskine.io.load_gridded_adcp(mooring=2)

# %%
subset = slice("2019-11-09", "2019-11-23")

# %%
aa = adcp.sel(time=subset)

# %%
test = aa.u.interpolate_na(dim="time", limit=20)

# %%
ax = test.gv.tplot()

# %%
ax = aa.u.gv.tplot()

# %% [markdown]
# We want the bandpass cutoff $c$ to be $c f < c^{-1} \omega_{SD}$, that is not to overlap with the semidiurnal tidal band.

# %%
f = gv.ocean.inertial_frequency(adcp.lat)
omega_sd = (2*np.pi)/(12.4*3600)

# %%
c = 1.06

# %%
tlow, thigh = niskine.calcs.determine_ni_band(bandwidth=c)
print(tlow, thigh)

# %%
(c*f) / ((1/c)*omega_sd)

# %% [markdown]
# Define a broader cutoff for testing.

# %%
c2 = 1.15

# %%
tlow, thigh = niskine.calcs.determine_ni_band(bandwidth=c2)
print(tlow, thigh)


# %%
def ni_bandpass_adcp(adcp, bandwidth=1.06, interpolate_gap_h=4):
    tlow, thigh = niskine.calcs.determine_ni_band(bandwidth=bandwidth)
    outu = adcp.u.copy()
    outu = outu.interpolate_na(
        dim="time", max_gap=np.timedelta64(interpolate_gap_h, "h")
    )
    outv = adcp.v.copy()
    outv = outv.interpolate_na(
        dim="time", max_gap=np.timedelta64(interpolate_gap_h, "h")
    )
    i = 0
    for g, aai in outu.groupby("z"):
        outu[i, :] = niskine.calcs.bandpass_time_series(aai.data, tlow, thigh, fs=6)
        i += 1
    i = 0
    for g, aai in outv.groupby("z"):
        outv[i, :] = niskine.calcs.bandpass_time_series(aai.data, tlow, thigh, fs=6)
        i += 1
    adcp["bpu"] = outu
    adcp["bpv"] = outv
    return adcp


# %% [markdown]
# reviewer asked for calculating WKB with full-depth averaged $N_0$. Let's see how much the factor changes...

# %%
m1lon, m1lat, m1depth = niskine.io.mooring_location(mooring=1)
n2, tz = niskine.clim.climatology_argo_woce(m1lon, m1lat, m1depth)
an2 = niskine.clim.interpolate_seasonal_data(adcp.time, n2)
adcp["n2"] = an2.interp_like(adcp)
adcp["N"] = np.sqrt(adcp.n2)
N0 = adcp.N.where((adcp.z<1200) & (adcp.z>300)).mean().item()

# %%
N0

# %%
N0new = adcp.N.mean().item()
N0new


# %% [markdown]
# Not much of a change. We'll apply this. Code in `niskine.clim.get_wkb_factors()` has been changed.

# %%
def calc_ni_eke(adcp):
    rho = 1025
    # load WKB normalization matrix
    wkb = niskine.clim.get_wkb_factors(adcp)
    # calculate NI EKE
    adcp["ni_eke"] = 0.5 * rho * ((wkb * adcp.bpu) ** 2 + (wkb * adcp.bpv) ** 2)
    return adcp


# %%
def calc_ni_eke_no_wkb(adcp):
    rho = 1025
    # calculate NI EKE
    adcp["ni_eke"] = 0.5 * rho * (adcp.bpu ** 2 + adcp.bpv ** 2)
    return adcp


# %%
a = ni_bandpass_adcp(adcp.copy())
a = calc_ni_eke(a)
a_no_wkb = ni_bandpass_adcp(adcp.copy())
a_no_wkb = calc_ni_eke_no_wkb(a_no_wkb)

# %%
a_wide = ni_bandpass_adcp(adcp.copy(), bandwidth=c2)
a_wide = calc_ni_eke(a_wide)

# %%
a2 = ni_bandpass_adcp(adcp2.copy())
a2 = calc_ni_eke(a2)

# %%
a.ni_eke.to_netcdf(cfg.data.ni_eke_m1)

# %%
a_no_wkb.ni_eke.to_netcdf(cfg.data.ni_eke_m1_no_wkb)

# %%
a2.ni_eke.to_netcdf(cfg.data.ni_eke_m2)

# %%
(a.bpu.sel(time=slice("2019-09", "2019-10"))**2).gv.plot(robust=True)

# %%
fig, ax = gv.plot.quickfig(w=8)
h = (
    a.ni_eke.resample(time="8h")
    .mean()
    .plot.contourf(
        ax=ax,
        extend="both",
        levels=np.arange(0.5, 4, 0.5),
        cmap="Spectral_r",
        antialiased=True,
        cbar_kwargs=dict(
            aspect=30,
            shrink=0.7,
            pad=0.01,
            label="NI EKE",
        ),
    )
)
for c in h.collections:
    c.set_rasterized(True)
    c.set_edgecolor("face")

ax = mld.gv.tcoarsen(n=30 * 24 * 2).gv.tplot(ax=ax, color="k", linewidth=1, alpha=0.6)
ax.invert_yaxis()
ax.set(xlabel="", ylabel="depth [m]", title="NI EKE M1", ylim=(1400, 0))
gv.plot.concise_date(ax)

# %%
fig, ax = gv.plot.quickfig(w=8)
h = (
    a_wide.ni_eke.resample(time="8h")
    .mean()
    .plot.contourf(
        ax=ax,
        extend="both",
        levels=np.arange(0.5, 4, 0.5),
        cmap="Spectral_r",
        antialiased=True,
        cbar_kwargs=dict(
            aspect=30,
            shrink=0.7,
            pad=0.01,
            label="NI EKE",
        ),
    )
)
for c in h.collections:
    c.set_rasterized(True)
    c.set_edgecolor("face")

ax = mld.gv.tcoarsen(n=30 * 24 * 2).gv.tplot(ax=ax, color="k", linewidth=1, alpha=0.6)
ax.invert_yaxis()
ax.set(xlabel="", ylabel="depth [m]", title="NI EKE M1 wide band", ylim=(1400, 0))
gv.plot.concise_date(ax)

# %%
(a_wide.ni_eke - a.ni_eke).resample(time="8h").mean().gv.plot(
    robust=True, cmap="RdBu_r"
)


# %%
def plot_3month_ni_ke(ts):
    fig, ax = gv.plot.quickfig(w=8, h=3, grid=True)
    h = (
        a_wide.ni_eke.sel(time=ts)
        .resample(time="7h")
        .mean()
        .plot.contourf(
            ax=ax,
            extend="both",
            levels=np.arange(0.5, 4, 0.5),
            cmap="Spectral_r",
            antialiased=True,
            cbar_kwargs=dict(
                aspect=30,
                shrink=0.7,
                pad=0.01,
                label="NI EKE",
            ),
        )
    )
    for c in h.collections:
        c.set_rasterized(True)
        c.set_edgecolor("face")

    mld_sel = mld.sel(time=ts).gv.tcoarsen(n=30 * 24 * 2)
    ax = mld_sel.gv.tplot(ax=ax, color="w", linewidth=2, alpha=1)
    ax = mld_sel.gv.tplot(ax=ax, color="k", linewidth=1, alpha=1)

    ax.invert_yaxis()
    ax.set(xlabel="", ylabel="depth [m]", title="NI EKE M1", ylim=(1400, 0))
    gv.plot.concise_date(ax)


# %%
tt = [slice("2019-05", "2019-09"), slice("2019-10", "2019-12"), slice("2020-01", "2020-03"), slice("2020-04", "2020-06"), slice("2020-07", "2020-10")]
for ts in tt:
    plot_3month_ni_ke(ts)

# %% [markdown]
# Is this pattern some kind of beating with the semidiurnal tide?

# %%
ts = slice("2020-01", "2020-03")
fig, ax = gv.plot.quickfig(w=8, grid=True)
h = (
    a_wide.ni_eke.sel(time=ts)
    .resample(time="7h")
    .mean()
    .plot.contourf(
        ax=ax,
        extend="both",
        levels=np.arange(0.5, 4, 0.5),
        cmap="Spectral_r",
        antialiased=True,
        cbar_kwargs=dict(
            aspect=30,
            shrink=0.7,
            pad=0.01,
            label="NI EKE",
        ),
    )
)
for c in h.collections:
    c.set_rasterized(True)
    c.set_edgecolor("face")

mld_sel = mld.sel(time=ts).gv.tcoarsen(n=30 * 24 * 2)
ax = mld_sel.gv.tplot(ax=ax, color="w", linewidth=2, alpha=1)
ax = mld_sel.gv.tplot(ax=ax, color="k", linewidth=1, alpha=1)

ax.invert_yaxis()
ax.set(xlabel="", ylabel="depth [m]", title="NI EKE M1", ylim=(1400, 0))
gv.plot.concise_date(ax)

# %%
# load wind work
wind_work = xr.open_dataarray(cfg.data.wind_work.niskine_m1)
wind_work.close()
wind_work_c = xr.open_dataarray(cfg.data.wind_work.niskine_m1_cumulative)
wind_work_c.close()
wind_stress = xr.open_dataarray(cfg.data.wind_work.niskine_m1_wind_stress)
wind_stress.close()

# load MLD
mld = xr.open_dataarray(cfg.data.ml.mld)
mld.close()

# %%
ni_eke = a.ni_eke
mldi = mld.interp_like(ni_eke)
mld_mask = ni_eke.z < mldi

# %%
(ni_eke.where(mld_mask).sum(dim="z") * 16).plot()

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 8),
                       constrained_layout=True, sharex=True)

wind_work.plot(ax=ax[0])
wind_work_c.plot(ax=ax[1])

(a.ni_eke.sum(dim="z") * 16 / 1e3).plot(ax=ax[2])
ax[2].set(ylabel="$\mathrm{EKE}_\mathrm{NI}$ [kJ/m$^2$]")

(ni_eke.where(mld_mask).sum(dim="z") * 16 / 1e3).plot(ax=ax[3])
gv.plot.concise_date(ax[3])
ax[3].set(ylabel="$\mathrm{EKE}_\mathrm{NI} (ML)$ [kJ/m$^2$]")

for axi in ax:
    gv.plot.axstyle(axi, grid=True)

# %%
fig, ax = gv.plot.quickfig(w=8)
h = (
    a2.ni_eke.resample(time="8h")
    .mean()
    .plot.contourf(
        ax=ax,
        extend="both",
        levels=np.arange(0.5, 4, 0.5),
        cmap="Spectral_r",
        antialiased=True,
        cbar_kwargs=dict(
            aspect=30,
            shrink=0.7,
            pad=0.01,
            label="NI EKE",
        ),
    )
)
for c in h.collections:
    c.set_rasterized(True)
    c.set_edgecolor("face")

ax = mld.gv.tcoarsen(n=30 * 24 * 2).gv.tplot(ax=ax, color="k", linewidth=1, alpha=0.6)
ax.invert_yaxis()
ax.set(xlabel="", ylabel="depth [m]", title="NI EKE M2", ylim=(1400, 0))
gv.plot.concise_date(ax)

# %% [markdown]
# I want to have a function that filters (low/band/high) an xarray DataArray in the time domain. It should
# - automatically be applied along time dimension
# - automatically detect sampling frequency
# - take cutoff periods in hours (instead of frequencies)
# - take the filter order as an optional parameter
# - return an xarray.DataArray

# %% [markdown]
# ## WKB stretching

# %%
wkb = niskine.clim.get_wkb_factors(a)

# %%
wkb.resample(time="1W").mean().gv.tplot()

# %%

# %%

# %% [markdown]
# ## interpolate gaps

# %% [markdown]
# How can we interpolate gaps near the surface such that we still get good near-surface NI velocities? Simply using `interpolate_na` along the time dimension may not do the trick.

# %%
ax = a.u.sel(time=slice('2020-02-01', '2020-02-08')).interpolate_na(dim='z').gv.tplot()
ax.set(ylim=(500, 0))
# gv.plot.png('gaps_simple ')

# %%
ax = a.u.sel(time=slice('2020-02-01', '2020-02-08')).interpolate_na(dim='time').gv.tplot()
ax.set(ylim=(500, 0))
# gv.plot.png('gaps_simple ')

# %% [markdown]
# I have objective mapping code from Jesse that may help here...
