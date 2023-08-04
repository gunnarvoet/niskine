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

# %%
gv.ocean.inertial_period(58);

# %% [markdown]
# # NISKINe Low-Mode NI Wave Fluxes

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "f9337f7b8cb73", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
os4 = niskine.mooring.OSNAPMooring(moorstr='UMM4')

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 3),
                       constrained_layout=True, sharey=True)
niskine.osnap.plot_mooring_setup(os4, ax=ax)
ax.invert_yaxis()

# %%
O = niskine.flux.Flux(mooring=os4, bandwidth=1.05, runall=True)

# %%
m1 = niskine.mooring.NISKINeMooring()

# %%
N = niskine.flux.Flux(mooring=m1, bandwidth=1.05, runall=True, climatology="ARGO")

# %%
fig, ax = gv.plot.quickfig(yi=True, h=3)
ax.plot(N.mooring.time, np.cumsum(N.Fu[:, 0])*3600/1e9, color='0.2')
ax.plot(N.mooring.time, np.cumsum(np.sum(N.Fu, axis=1))*3600/1e9, label='F$_\mathregular{u}$')
ax.plot(N.mooring.time, np.cumsum(N.Fv[:, 0])*3600/1e9, color='0.2')
ax.plot(N.mooring.time, np.cumsum(np.sum(N.Fv, axis=1))*3600/1e9, label='F$_\mathregular{v}$')
ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]',
       title='NISKINe M1 Near-Inertial Energy Flux')
ax.legend()
gv.plot.concise_date()
ax.grid()
# niskine.io.png("niskine_m1_ni_low_mode_flux_all", subdir="low-mode-fluxes")

# %% [markdown]
# Check out modes based on Argo climatology:

# %% [markdown]
# These are time-mean vertical and horizontal modes

# %%
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
h0 = N.modes.vmodes.mean(dim="time").plot(hue="mode", y="z", yincrease=False, ax=ax[0], )
h1 = N.modes.hmodes.mean(dim="time").plot(hue="mode", y="z", yincrease=False, ax=ax[1], )

# %% [markdown]
# Here is the seasonal evolution of the modes

# %%
out = N.modes.vmodes.plot(col="mode", yincrease=False)
for axi in out.axs[0]:
    axi.set(xlabel="")
# out.fig.suptitle("vmodes")

# %%
out = N.modes.hmodes.plot(col="mode", yincrease=False)
for axi in out.axs[0]:
    axi.set(xlabel="")

# %%
ax = N.Tz.gv.tplot(vmin=-0.03)

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_up_one_time_step(N, ti=6000)

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_eta_modes_one_time_step(N, 6000)

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for axi, (g, ni) in zip(ax[:-1], N.up.groupby("mode")):
    ni.gv.tplot(ax=axi, cmap="RdBu_r", vmin=-0.3, vmax=0.3)

N.up.sum(dim="mode").gv.tplot(ax=ax[-1], cmap="RdBu_r", vmin=-0.5, vmax=0.5)

# %%
N.up.sum(dim="mode").gv.tplot(cmap="RdBu_r", )

# %%
N.pp.isel(mode=0).gv.tplot(cmap="RdBu_r")

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for axi, (g, ni) in zip(ax[:-1], N.up.groupby("mode")):
    ni.gv.tplot(ax=axi, cmap="RdBu_r", vmin=-0.3, vmax=0.3)

N.up.sum(dim="mode").gv.tplot(ax=ax[-1], cmap="RdBu_r", vmin=-0.5, vmax=0.5)

# %%

# %% [markdown]
# The mode fits for OSNAP look pretty okay, at least the ones for the horizontal or velocity modes.

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_up_one_time_step(O, ti=800)

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_eta_modes_one_time_step(O, 1000)

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)
for axi, (g, ni) in zip(ax[:-1], O.up.groupby("mode")):
    ni.gv.tplot(ax=axi, cmap="RdBu_r", vmin=-0.05, vmax=0.05)

O.up.sum(dim="mode").gv.tplot(ax=ax[-1], cmap="RdBu_r", vmin=-0.1, vmax=0.1)

# %%
O.up.sum(dim="mode").gv.tplot(cmap="RdBu_r", )

# %%

# %%

# %%

# %% [markdown]
# Pick a shorter time range for testing and development.

# %%
s = m1.shorten(slice("2020-03", "2020-04"))

# %%
s

# %% [markdown]
# Play with the NI flux calculation here. We use the shorter time series to speed things up a bit.

# %%
test = niskine.flux.Flux(mooring=s, bandwidth=1.03, runall=True, climatology="ARGO")

# %%
# test.background_gradients?

# %%

# %%

# %%

# %% [markdown]
# Now that the flux calculation runs we want to make sure that the calculation is doing the right thing. Things to look for:
# - Is the velocity mode projection okay?
# - How about the mode projection for $\eta$?
#
# I don't quite know how to evaluate the mode projections.

# %%
binbpu, binbpv = niskine.flux.downsample_adcp_data(test.mooring.adcp)

# %%
binbpu.isel(z_bins=20).plot()

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_up_one_time_step(test, ti=1000)

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_eta_modes_one_time_step(test, 1000)

# %%
np.log10(N.N2s).plot()

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_up_one_time_step(O, ti=1100)

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_eta_modes_one_time_step(O, 1500)

# %%

# %%

# %%

# %%

# %%

# %%
test.flux.fx_ni.isel(mode=0).plot()
test.flux.fy_ni.isel(mode=0).plot()
gv.plot.concise_date()

# %%
test.flux

# %% [markdown]
# Okay, now we have the NISKINe M1 mooring data in the right format for the flux calculation.

# %%
O = niskine.flux.Flux(mooring=os4, bandwidth=1.05, runall=True)

# %%
os3.cm = os3.cm.expand_dims("nomz")
O3 = niskine.flux.Flux(mooring=os3, bandwidth=1.05, runall=True)

# %%
N = niskine.flux.Flux(mooring=m1, bandwidth=1.03, runall=True, climatology="ARGO")

# %%
O.background_gradients()

# %%
N.background_gradients()

# %%
fig, ax = gv.plot.quickfig(w=3)
N.N2.plot(y="z")
O.N2.plot(y="z")
ax.invert_yaxis()

# %%
N.find_modes()
O.find_modes()

# %%
fig, ax = gv.plot.quickfig(w=3)
N.modes.vmodes.plot(hue="mode", y="z")
O.modes.vmodes.plot(hue="mode", y="z")
ax.invert_yaxis()

# %%
O.bandpass()

# %%
N.bandpass()

# %%
N.mooring.adcp.bpv.sel(time=slice("2020-04", "2020-06")).gv.tplot()

# %%
O.eta_modes()

# %%
zz = np.tile(N.mooring.ctd.z, (len(N.mooring.ctd.time), 1))

N.mooring.ctd["zz"] = (("z", "time"), zz.transpose())

N.mooring.time = N.mooring.adcp.time.data

# %%
N.eta_modes()

# %%
O.vel_modes()

# %%
N.vel_modes()

# %%
O.calc_pp()

# %%
N.calc_pp()

# %%
O.flux_calcs()

# %%
N.flux_calcs()

# %%
(N.vp.sel(mode=1)*N.pp.sel(mode=1)).gv.tplot()

# %%
fig, ax = gv.plot.quickfig()
N.mooring.adcp.xducer_depth.plot(hue="adcp")
gv.plot.concise_date(ax)

# %%
N.Fu.sel(mode=1).plot()
N.Fu.sel(mode=2).plot()
N.Fu.sel(mode=3).plot()

# %%
N.Fv.sel(mode=1).plot()
N.Fv.sel(mode=2).plot()
N.Fv.sel(mode=3).plot()

# %%
O.Fu.sel(mode=1).plot()
O.Fu.sel(mode=2).plot()
O.Fu.sel(mode=3).plot()

# %%
O.Fv.sel(mode=1).plot()
O.Fv.sel(mode=2).plot()
O.Fv.sel(mode=3).plot()

# %% [markdown]
# Let's see if we get a result that is similar to before

# %%
fig, ax = gv.plot.quickfig(yi=True, h=3)
ax.plot(os4.time, np.cumsum(O.Fu[:, 0])*3600/1e9, color='0.2')
ax.plot(os4.time, np.cumsum(np.sum(O.Fu, axis=1))*3600/1e9, label='F$_\mathregular{u}$')
ax.plot(os4.time, np.cumsum(O.Fv[:, 0])*3600/1e9, color='0.2')
ax.plot(os4.time, np.cumsum(np.sum(O.Fv, axis=1))*3600/1e9, label='F$_\mathregular{v}$')
ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]',
       title='OSNAP MM4 Near-Inertial Energy Flux')
ax.legend()
gv.plot.concise_date()
ax.grid()
niskine.io.png("osnap_mm4_ni_low_mode_flux", subdir="low-mode-fluxes")

# %%
fig, ax = gv.plot.quickfig(yi=True, h=3)
ax.plot(os3.time, np.cumsum(O3.Fu[:, 0])*3600/1e9, color='0.2')
ax.plot(os3.time, np.cumsum(np.sum(O3.Fu, axis=1))*3600/1e9, label='F$_\mathregular{u}$')
ax.plot(os3.time, np.cumsum(O3.Fv[:, 0])*3600/1e9, color='0.2')
ax.plot(os3.time, np.cumsum(np.sum(O3.Fv, axis=1))*3600/1e9, label='F$_\mathregular{v}$')
ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]',
       title='OSNAP MM3 Near-Inertial Energy Flux')
ax.legend()
gv.plot.concise_date()
ax.grid()
niskine.io.png("osnap_mm3_ni_low_mode_flux", subdir="low-mode-fluxes")

# %%
fig, ax = gv.plot.quickfig(yi=True, h=3)
ax.plot(N.mooring.time, np.cumsum(N.Fu[:, 0])*3600/1e9, color='0.2')
ax.plot(N.mooring.time, np.cumsum(np.sum(N.Fu, axis=1))*3600/1e9, label='F$_\mathregular{u}$')
ax.plot(N.mooring.time, np.cumsum(N.Fv[:, 0])*3600/1e9, color='0.2')
ax.plot(N.mooring.time, np.cumsum(np.sum(N.Fv, axis=1))*3600/1e9, label='F$_\mathregular{v}$')
ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]',
       title='NISKINe M1 Near-Inertial Energy Flux')
ax.legend()
gv.plot.concise_date()
ax.grid()
niskine.io.png("niskine_m1_ni_low_mode_flux_all", subdir="low-mode-fluxes")

# %%
p_limit=303
np.sum(N.mooring.adcp.xducer_depth.sel(adcp=3109)>p_limit)/np.sum(N.mooring.adcp.xducer_depth.sel(adcp=3109)<p_limit)

# %%
p_limit=303

Fv_nk = N.Fv.where(N.mooring.adcp.xducer_depth.sel(adcp=3109)<p_limit).copy().drop("adcp")
Fu_nk = N.Fu.where(N.mooring.adcp.xducer_depth.sel(adcp=3109)<p_limit).copy().drop("adcp")
Fv_nk_m = np.cumsum(Fv_nk, axis=0)*3600/1e9
Fu_nk_m = np.cumsum(Fu_nk, axis=0)*3600/1e9
Fv_nk_int = np.cumsum(np.sum(Fv_nk, axis=1))*3600/1e9
Fu_nk_int = np.cumsum(np.sum(Fu_nk, axis=1))*3600/1e9
fig, ax = gv.plot.quickfig(h=4, w=7, fs=12)
Fu_nk_m.sel(mode=1).plot(label='F$_\mathregular{east}$')
Fu_nk_int.plot(color="k")
Fv_nk_m.sel(mode=1).plot(label='F$_\mathregular{north}$')
Fv_nk_int.plot(color="k")
ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]',)
#        title='NISKINe M1 Near-Inertial Energy Flux')
gv.plot.concise_date()
ax.legend(fontsize=12)
ax.grid()
from matplotlib import ticker
ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
niskine.io.png("niskine_m1_ni_low_mode_flux", subdir="low-mode-fluxes")

# %% [markdown]
# Okay, so NISKINE is about an order of magnitude larger than OSNAP. How much flux do we see on average? These should be in W/m.

# %%
N.Fu.mean()

# %%
N.Fv.mean()

# %% [markdown]
# I don't understand why average OSNAP fluxes are so small. On the map I showed in 2020 they were about 50 W/m$^2$.

# %%
O.Fv.mean()

# %%
O.Fu.mean()

# %% [markdown]
# Actually, the OSNAP fluxes seem rather small compared to what Matthew shows in his 2003 Nature paper and the NISKINe fluxes are more in line with what he finds there...

# %%
fig, ax = gv.plot.quickfig(yi=True, w=3)
O.mooring.eta.plot.hist(bins=np.arange(-40, 41, 1), color='0.2');

# %%
fig, ax = gv.plot.quickfig(yi=True, w=3)
N.mooring.eta.plot.hist(bins=np.arange(-40, 41, 1), color='0.2');

# %%
N.mooring.adcp.bpu.sel(time=slice("2020-05", "2020-07")).gv.tplot()

# %%
N.vp.sel(mode=1).sel(time=slice("2020-05", "2020-07")).gv.tplot()

# %%
N.pp.sel(mode=1).sel(time=slice("2020-05", "2020-07")).gv.tplot()

# %%
fig, ax = gv.plot.quickfig()
N.mooring.adcp.xducer_depth.sel(time=slice("2020-04", "2020-07")).plot(hue="adcp")
gv.plot.concise_date(ax)
ax.grid()
ax.invert_yaxis()

# %%
fig, ax = gv.plot.quickfig(yi=True, h=3)
ax.plot(N.Fv.time.sel(time=slice("2020-04", "2020-07")), np.cumsum(np.sum(N.Fv, axis=1)).sel(time=slice("2020-04", "2020-07"))*3600/1e9, label='F$_\mathregular{v}$')
ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]',
       title='NISKINe M1 Near-Inertial Energy Flux')
ax.legend()
ax.grid()
gv.plot.concise_date()

# %%
ax = (N.vp.sel(mode=1).sel(time=slice("2020-05", "2020-07"))*N.pp.sel(mode=1)).sel(time=slice("2020-05", "2020-07")).gv.tplot(cmap="RdBu_r")
ax.grid()

# %%
(N.up.sel(mode=1).sel(time=slice("2020-05", "2020-07"))*N.pp.sel(mode=1)).sel(time=slice("2020-05", "2020-07")).gv.tplot(cmap="RdBu_r")

# %%

# %%

# %%

# %%
fig, ax = gv.plot.quickfig(yi=True)
absF = xr.DataArray(data=np.sqrt(O.Fu[:, 0]**2 + O.Fv[:,0]**2),
                    coords=dict(time=O.mooring.time))
ax.plot(os4.time, absF, label='hourly estimate')
absF.rolling(dict(time=48)).mean().plot(label='48hr rolling mean')
ax.set(ylabel='$|F_\mathregular{NI}|$ [W/m]')
gv.plot.concise_date()
ax.legend()
# ax.set(yscale='log')
ax.set(title='OSNAP MM4 NI flux magnitude')
# gv.plot.png('mm4_ni_flux_magnitude_time_series')

# %%
fig, ax = gv.plot.quickfig(yi=True)
absF = xr.DataArray(data=np.sqrt(Fu[:, 0]**2, Fv[:,0]**2),
                    coords=dict(time=os4.time), dims=['time'])
ax.plot(os4.time, absF, label='hourly estimate')
absF.rolling(dict(time=48)).mean().plot(label='48hr rolling mean')
ax.set(ylabel='$|F_\mathregular{NI}|$ [W/m]')
gv.plot.concise_date()
ax.legend()
# ax.set(yscale='log')
ax.set(title='OSNAP MM4 NI flux magnitude')
gv.plot.png('mm4_ni_flux_magnitude_time_series')

# %%

# %%
niskine.flux.plot_eta_modes_time_series(N)

# %%
niskine.flux.plot_up_modes_time_series(N)

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_up_one_time_step(N, ti=9300)

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_eta_modes_one_time_step(N, 5000)

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Climatological Data

# %% [markdown]
# The goal is to have one DataArray with $N^2$ and coordinate `z`, in the case of using the Argo climatology also `time`. Also one DataArray with $T_z$.

# %% [markdown]
# WOCE

# %%
N2, Tz = niskine.clim.climatology_woce(lon, lat, bottom_depth)

# %% [markdown] janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "76a8c910556a48", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
# WOCE Argo Climatology

# %%
N2a, Tza = niskine.clim.climatology_argo_woce(lon, lat, bottom_depth)

# %% [markdown]
# ## Band-pass Filter

# %% [markdown]
# We need to band-pass temperature and velocity.
#
# - In-situ or potential temperature? We should use potential temperature as the gradient was calculated that way as well.
#
# - Can't feed NaNs into the filtering routine. Need to interpolate over little blocks of NaNs and break into chunks if there are larger time spans of missing data.

# %% [markdown]
# Doing this in `osnap` now.

# %% [markdown]
# - First break up into chunks
# - Then interpolate over NaNs
# - Band-pass
# - Stitch back together

# %%
f = gv.ocean.inertial_frequency(lat=lat)
t = gv.ocean.inertial_period(lat=lat) * 24
thigh = 1/1.05*t
tlow = 1.05*t
print('Filter parameters: {:1.2f}h {:1.2f}h {:1.2f}h'.format(tlow, t, thigh))

# %% [markdown]
# Make sure they all fall within sensible ranges

# %%
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 3),
                       constrained_layout=True)
for axi, vi, lbl in zip(ax, [os4.cm.bpu, os4.cm.bpv, os4.adcp.bpu, os4.adcp.bpv],
                   ['cm u', 'cm v', 'adcp u', 'adcp v']):
    vi.plot.hist(bins=np.arange(-0.07, 0.072, 0.002), ax=axi, color='0.2',
                 label=lbl);
    axi.set(title=lbl)

# %% [markdown]
# ### test bandpass filtering

# %%
test = gv.signal.bandpassfilter(scipy.signal.detrend(th1[1:].data),
                                lowcut=1/tlow,
                                highcut=1/thigh,
                                fs=1, order=2)

# %%
bp = xr.DataArray(test, coords=dict(time=th1.time[1:]), dims=['time'])

# %%
tsel = slice('2014-12-11', '2014-12-19')

# %%
fig, ax = gv.plot.quickfig()
(th1-th1.mean(dim='time')).sel(time=tsel).plot(ax=ax, color='0.5')
bp.sel(time=tsel).plot(ax=ax, color='0.2');

# %%
tsel = slice('2014-09', '2015-12')

# %%
fig, ax = gv.plot.quickfig()
(th1-th1.mean(dim='time')).sel(time=tsel).plot(ax=ax, color='0.5')
bp.sel(time=tsel).plot(ax=ax, color='0.2');

# %% [markdown]
# Let's construct a sample time series with tidal and near-inertial components and filter them to see what we get.

# %%
faketime = os4.time.copy()
n = len(faketime)

omega_ni = 2*np.pi/t
omega_m2 = 2*np.pi/12.4

faken = np.arange(0, n)
fakeni = np.sin(omega_ni * faken) + np.random.random(n)
fakem2 = np.sin(omega_m2 * faken) + np.random.random(n)
fake = fakeni + fakem2

# %%
fig, ax = gv.plot.quickfig(h=3)
ax.plot(faketime[:100], fakeni[:100])
ax.plot(faketime[:100], fake[:100])
ax.plot(faketime[:100], fakem2[:100])
gv.plot.concise_date(ax)

# %%
test = gv.signal.bandpassfilter(fake,
                                lowcut=1/tlow,
                                highcut=1/thigh,
                                fs=1, order=2)

# %%
fake = xr.DataArray(fake, coords=dict(time=faketime), dims=['time'])
bpfake = xr.DataArray(test, coords=dict(time=faketime), dims=['time'])

# %%
tsel = slice('2014-12-11', '2014-12-19')

# %%
fig, ax = gv.plot.quickfig()
fake.sel(time=tsel).plot(ax=ax, color='0.5')
bpfake.sel(time=tsel).plot(ax=ax, color='0.2');

# %% [markdown]
# ## Calculate Eta

# %% [markdown]
# Do not use nominal depth of the instrument but calculate depth from pressure on the CTD for each time step.

# %% [markdown]
# $$
# \eta(z_j, t) = \frac{T(z_j, t)}{T_z(z_j)}
# $$

# %% [markdown]
# There is quite a bit of mooring knockdown which will bias these estimates. Could we correct for the knockdowns by subtracting the signal that is due to vertical motion through the background stratification?

# %%
os4.ctd.zz.plot()

# %%
os4.calc_eta(Tz)

# %%
fig, ax = gv.plot.quickfig(yi=True, w=3)
os4.eta.plot.hist(bins=np.arange(-40, 41, 1), color='0.2');

# %%
os4.eta.sel(time='2015-01').plot(y='nomz', yincrease=False);

# %% [markdown]
# ## Mode Fits

# %% [markdown]
# I did some mode fits last year - I think this was when I was trying to calculate tidal fluxes for the FLEAT lee wave study. Look into that! Turns out that's Matlab code based on `sw_vmodes`.

# %% [markdown]
# Let's calculate some modes - I found code on [github](https://github.com/UBC-MOAD/AIMS-Workshop/tree/master/dynmodes) in a [collection of GFD exercises](https://www.eoas.ubc.ca/~sallen/AIMS-workshop/Ex2-DynamicModes.html) by Doug Latornell. See also this [notebook](https://github.com/UBC-MOAD/AIMS-Workshop/blob/master/dynmodes/dynmodes.ipynb).

# %%
import dynmodes

# %%
wmodes, pmodes, rmodes, ce = dynmodes.dynmodes(N2, N2dep, 3)

# %%
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(7.5, 5),
                       constrained_layout=True, sharey=True)
opts = dict(color='0.2')
ax[0].plot(N2, N2dep, **opts)
for i in range(3):
    ax[1].plot(wmodes[i, :], N2dep, **opts)
    ax[2].plot(pmodes[i, :], N2dep, **opts)
    ax[3].plot(rmodes[i, :], N2dep, **opts)
ax[0].invert_yaxis()

# %% [markdown]
# Does the amplitude of the mode fits matter? The matlab routine `sw_vmodes` normalizes each mode to 1. Since we are projecting data on the modes, only the shape should matter?

# %% [markdown]
# ### Compare to Matlab sw_vmodes result

# %%
matres = gv.io.loadmat('sw_vmodes_result.mat')

# %%
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(7.5, 5),
                       constrained_layout=True, sharey=True)
opts = dict(color='0.2')
opts1 = dict(color='lightblue')
ax[0].plot(N2, N2dep, **opts)
for i in range(3):
    ax[1].plot(matres['vmodes'].transpose()[i+1, :]/5, N2dep, **opts)
    ax[2].plot(matres['hmodes'].transpose()[i+1, :]/1000, N2dep, **opts)
    ax[1].plot(wmodes[i, :], N2dep, **opts1)
    ax[2].plot(pmodes[i, :], N2dep, **opts1)
#     ax[3].plot(rmodes[i, :], N2dep, **opts)
ax[0].invert_yaxis()

# %% [markdown]
# Use the Matlab mode fits for now.

# %%
wmodes_old = wmodes.copy()
pmodes_old = pmodes.copy()

# %%
wmodes = matres['vmodes'].transpose()[1:4, :]
pmodes = matres['hmodes'].transpose()[1:4, :]

# %% [markdown]
# ## Project Observations on Modes

# %% [markdown]
# Let's start out with an $\eta$ profile and regress it on the first two $w$ or vertical modes. These are the `vmodes` in the Matlab package.

# %%
eta = os4.eta.isel(time=3000)

# %%
fig, ax = gv.plot.quickfig(w=3.5)
eta.plot(y='nomz', yincrease=False, ax=ax)

# %%
vme = interp1d(N2dep, wmodes)(eta.nomz)

# %%
vme.shape

# %%
fig, ax = gv.plot.quickfig(w=3.5)
for vmei in vme:
    ax.plot(vmei, eta.nomz, color='0.2')


# %%
y = eta.data
X = vme
X = X.transpose()

# %%
X = np.c_[X, np.ones(X.shape[0])]

# %%
beta_hat = np.linalg.lstsq(X,y, rcond=None)[0]

# %%
beta_hat = osnap.project_on_modes(eta.data, eta.nomz.data, wmodes, N2dep)

# %%
vme.shape

# %%
fig, ax = gv.plot.quickfig(fgs=(5,4))
eta.plot(y='nomz', yincrease=False, ax=ax)
for i in range(3):
    ax.plot(np.dot(vme[i, :], beta_hat[i]), eta.nomz, color='0.5')
ax.plot(np.dot(X, beta_hat), eta.nomz, color='0.2', lw=2)
gv.plot.xsym()

# %% [markdown]
# Project each time stamp.

# %%
eta = os4.eta.copy()
etaz = eta.nomz.data
beta_hat = np.zeros((len(eta.time), 4))*np.nan
for i, (g, etai) in enumerate(eta.groupby('time')):
    ni = np.flatnonzero(np.isfinite(etai))
    beta_hat[i,:] = osnap.project_on_modes(etai.data[ni], etaz[ni], wmodes, N2dep)

# %% [markdown]
# Project velocities

# %%
os4.cm['zz'] = (['time', 'nomz'], -1*gsw.z_from_p(os4.cm.p.data, os4.lat))

# %%
fig, ax = gv.plot.quickfig()
os4.adcp.bpu.plot(y='z', yincrease=False)

# %% [markdown]
# We need to downsample the ADCP data, it looks like the projection happens mostly with the high data density in the upper 500m.

# %%
zbins = np.arange(0, 800, 200)
zbinlabels = np.arange(100, 700, 200)
binbpu = os4.adcp.bpu.groupby_bins('z', bins=zbins, labels=zbinlabels).mean()
binbpv = os4.adcp.bpv.groupby_bins('z', bins=zbins, labels=zbinlabels).mean()

# %%
fig, ax = gv.plot.quickfig()
binbpv.plot(y='z_bins', yincrease=False)


# %%
def combine_adcp_cm_one_timestep(mooring, adcpu, adcpz, time):
    # select and downsample ADCP data, then combine with cm
    au = adcpu.isel(time=time).data
    az = adcpz.data
    u = np.concatenate((au, mooring.cm.bpu.isel(time=time).data))
    z = np.concatenate((az, mooring.cm.zz.isel(time=time).data))
    ni = np.isfinite(u)
    return u[ni], z[ni]


# %%
n = len(os4.time)
betau_hat = np.zeros((len(eta.time), 4))*np.nan
for i in range(n):
    u, z = combine_adcp_cm_one_timestep(os4, binbpu, binbpu.z_bins, i)
    betau_hat[i,:] = osnap.project_on_modes(u, z, pmodes, N2dep)

# %%
n = len(os4.time)
betav_hat = np.zeros((len(eta.time), 4))*np.nan
for i in range(n):
    v, z = combine_adcp_cm_one_timestep(os4, binbpv, binbpv.z_bins, i)
    betav_hat[i,:] = osnap.project_on_modes(v, z, pmodes, N2dep)

# %%
timei=2100
v, z = combine_adcp_cm_one_timestep(os4, binbpv, binbpv.z_bins, timei)
fig, ax = gv.plot.quickfig(fgs=(5,4))
vme = interp1d(N2dep, pmodes, bounds_error=False)(z)
ax.plot(v, z, 'bo')
for i in range(3):
    ax.plot(np.dot(vme[i, :], betav_hat[timei, i]), z, color='0.5')
X = vme.transpose()
X = np.c_[X, np.ones(X.shape[0])]
ax.plot(np.dot(X, betav_hat[timei]), z, color='0.2', lw=2)
tmp = np.zeros((len(z), 3))
# sum of the modes (missing the barotropic component)
for i in range(3):
    tmp[:, i] = np.dot(vme[i, :], betav_hat[timei, i])
ax.plot(np.sum(tmp, axis=1), z, color='r')
gv.plot.xsym()

# %%
timei=2400
v, z = combine_adcp_cm_one_timestep(os4, binbpv, binbpv.z_bins, timei)
fig, ax = gv.plot.quickfig(fgs=(5,4))
vme = interp1d(N2dep, pmodes, bounds_error=False)(z)
ax.plot(v, z, 'bo')
for i in range(3):
    ax.plot(np.dot(vme[i, :], betav_hat[timei, i]), z, color='0.5')
X = vme.transpose()
X = np.c_[X, np.ones(X.shape[0])]
ax.plot(np.dot(X, betav_hat[timei]), z, color='0.2', lw=2)
tmp = np.zeros((len(z), 3))
# sum of the modes (missing the barotropic component)
for i in range(3):
    tmp[:, i] = np.dot(vme[i, :], betav_hat[timei, i])
ax.plot(np.sum(tmp, axis=1), z, color='r')
gv.plot.xsym()


# %% [markdown]
# ## Calculate pressure perturbation

# %% [markdown]
# Following Kunze et al. (2002) the pressure perturbation per mode, $p^\prime_i$, can be calculated by integrating buoyancy $b^\prime = \overline{N}^2(z) \zeta(z)$ with depth as
# $$
# \begin{align}
# p^\prime (z) &=& - \int_z^0 b^\prime(z^\prime) dz^\prime + p^\prime (0)\\
#              &=& - \int_z^0 b^\prime(z^\prime) dz^\prime + \frac{1}{H} \int_{-H}^0\int_z^0 b^\prime (z^\prime) dz^\prime dz\\
#              &=& \int_z^0 \overline{N}^2(z^\prime) \zeta(z^\prime)\, dz^\prime -
#              \frac{1}{H} \int_{-H}^0\int_z^0 \overline{N}^2 (z^\prime) \zeta (z^\prime)\, dz^\prime\, dz
# \end{align}
# $$

# %% [markdown]
# Note: This is what Kunze et al. call the reduced pressure anomaly, i.e. $p^\prime = P/\rho_0$.

# %% [markdown]
# How to do best the integration? Need to calculate $\Delta z$ for $N^2$ depths. Then sum up piecewise from the top.

# %%
def pp_per_mode(beta_hat, wmodes, N2, N2dep):
    # Insert zero into N2 depth vector at the top
    tmp = np.insert(N2dep, 0, 0)
    # Calculate delta z for N2 depth vector
    N2dz = np.diff(tmp)
    # Pprime will have shape (ntime, len(N2), nmodes)
    pp = np.full((len(os4.time), len(N2), 3), np.nan)
    for i in range(len(os4.time)):
        for j in range(3):
            pp[i, :, j] = np.cumsum(np.dot(wmodes[j, :], beta_hat[i, j])
                                    * N2
                                    * N2dz)
            pp[i, :, j] -= np.mean(pp[i, :, j])
    return pp


# %%
pp = pp_per_mode(beta_hat, wmodes, N2, N2dep)

# %%
fig, ax = gv.plot.quickfig(w=3.5)
for i in range(3):
    ax.plot(pp[1000, :, i], N2dep, color='0.2')

# %% [markdown]
# These look so small. We are calculating $\int N^2 \eta dz$. Let's make sure we have the units right here.
# $$N^2=s^{-2}$$
# $$\eta = m$$
# $$dz = m$$

# %% [markdown]
# This is m$^2$/s$^2$. For pressure, we want N/m$^2$, thus we are missing a factor in units of kg/m$^3$. Must be density.

# %% [markdown]
# Actually, turns out what we are calculating here is *reduced pressure anomaly* $p^\prime  = P/\rho_0$ (Kunze et al., 2002). This has then units of m$^2$/s$^2$. However, to calculate an energy flux, we then need to multiply with $\rho_0$ somwhere.

# %% [markdown]
# ## Calculate u', v' time series per mode

# %%
pmodes.shape


# %%
def up_per_mode(betau_hat, betav_hat, pmodes):
    # uprime will have shape (ntime, len(N2), nmodes)
    up = np.full((len(os4.time), pmodes.shape[1], 3), np.nan)
    vp = up.copy()
    for i in range(len(os4.time)):
        for j in range(3):
            up[i, :, j] = np.dot(pmodes[j, :], betau_hat[i, j])
            vp[i, :, j] = np.dot(pmodes[j, :], betav_hat[i, j])
    return up, vp


# %%
up, vp = up_per_mode(betau_hat, betav_hat, pmodes)

# %%
up.shape

# %%
ti = 2100
fig, ax = gv.plot.quickfig()
# for i in range(3):
#     ax.plot(up[1000, :, i], N2dep)
ax.plot(np.sum(vp[ti, :, :2], axis=1), N2dep, color='k')
os4.adcp.bpv.isel(time=ti).plot(y='z')
os4.cm.bpv.isel(time=ti).plot(y='nomz', marker='o')

# %%
fig, ax = gv.plot.quickfig()
h = ax.pcolormesh(os4.time, N2dep, up[:,:,0].transpose(), cmap='RdBu_r',
                 vmin=-0.07, vmax=0.07)
plt.colorbar(h)
gv.plot.concise_date()

# %%
fig, ax = gv.plot.quickfig(yi=True)
ax.hist(np.reshape(up, (-1, 1)), bins=np.arange(-0.1, 0.102, 0.002 ));
# ax.set(yscale='log')

# %% [markdown]
# ## Calculate flux

# %% [markdown]
# Near-inertial energy flux for each mode is calculated as
# $$ F_i(z) = \rho_0 \left< u_i p^\prime_i \right>
# $$
# in units of W/m$^2$. The factor $\rho_0$ is needed as $p^\prime$ calculated from $\eta$ is reduced pressure.

# %%
rho0 = 1025
Fuz = rho0 * up * pp
Fvz = rho0 * vp * pp

# %% [markdown]
# What are average values going into the flux calculation?

# %%
np.mean(np.absolute(up))

# %%
np.mean(np.absolute(pp*1025))

# %%
0.01 * 1 * 3000 / 1e3

# %% [markdown]
# Depth-integrated flux. Matthew shows time-mean depth-integrated fluxes of $\mathcal{O}(1)$ kW/m. This translates into an annual energy input of $\mathcal{O}(10)$ GJ/m.

# %%
# Insert zero into N2 depth vector at the top
tmp = np.insert(N2dep, 0, 0)
# Calculate delta z for N2 depth vector
N2dz = np.diff(tmp)
N2dz = np.reshape(N2dz, (1, -1, 1))

# %%
Fu = np.sum(Fuz*N2dz, axis=1)
Fv = np.sum(Fvz*N2dz, axis=1)

# %%
fig, ax = gv.plot.quickfig(yi=True)
absF = xr.DataArray(data=np.sqrt(Fu[:, 0]**2, Fv[:,0]**2),
                    coords=dict(time=os4.time), dims=['time'])
ax.plot(os4.time, absF, label='hourly estimate')
absF.rolling(dict(time=48)).mean().plot(label='48hr rolling mean')
ax.set(ylabel='$|F_\mathregular{NI}|$ [W/m]')
gv.plot.concise_date()
ax.legend()
# ax.set(yscale='log')
ax.set(title='OSNAP MM4 NI flux magnitude')
gv.plot.png('mm4_ni_flux_magnitude_time_series')

# %%
fig, ax = gv.plot.quickfig(yi=True, h=3)
ax.plot(os4.time, np.cumsum(Fu[:, 0])*3600/1e9, color='0.2')
ax.plot(os4.time, np.cumsum(np.sum(Fu, axis=1))*3600/1e9, label='F$_\mathregular{u}$')
ax.plot(os4.time, np.cumsum(Fv[:, 0])*3600/1e9, color='0.2')
ax.plot(os4.time, np.cumsum(np.sum(Fv, axis=1))*3600/1e9, label='F$_\mathregular{v}$')
ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]',
       title='OSNAP MM4 Near-Inertial Energy Flux')
ax.legend()
gv.plot.concise_date()
gv.plot.png('mm4_ni_flux_time_series')

# %%
print(np.mean(Fu[:,0]),  np.mean(Fv[:,0]))

# %%
print(np.mean(np.sum(Fu, axis=1)),  np.mean(np.sum(Fv, axis=1)))

# %%
mFu, mFv = np.mean(np.sum(Fu, axis=1)),  np.mean(np.sum(Fv, axis=1))

# %% [markdown]
# ## Run calculation in functional form.

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "f9337f7b8cb73", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
os4 = osnap.mooring(osnap_path, moorstr='UMM4')

# %%
Fu, Fv = osnap.near_inertial_energy_flux(os4, bandwidth=1.05, N2=N2, Tz=Tz)

# %%
np.mean(Fv, axis=0)

# %%
fig, ax = gv.plot.quickfig(yi=True, h=3)
ax.plot(os4.time, np.cumsum(Fu[:, 0])*3600/1e9, color='0.2')
ax.plot(os4.time, np.cumsum(np.sum(Fu, axis=1))*3600/1e9, label='F$_\mathregular{u}$')
ax.plot(os4.time, np.cumsum(Fv[:, 0])*3600/1e9, color='0.2')
ax.plot(os4.time, np.cumsum(np.sum(Fv, axis=1))*3600/1e9, label='F$_\mathregular{v}$')
ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]',
       title='OSNAP MM4 Near-Inertial Energy Flux')
ax.legend()
gv.plot.concise_date()
# gv.plot.png('mm4_ni_flux_time_series')

# %% [markdown]
# ## PDF of NI Flux

# %%
import matplotlib.colors as colors

# %%
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection="polar")
Fuv = Fu[:, 0] + 1j * Fv[:, 0]
# Fuv = mFu + 1j * mFv
Fmag, Fdir = np.absolute(Fuv), np.angle(Fuv)
Fdir = Fdir - np.pi / 2
Fdir[Fdir < 0] = Fdir[Fdir < 0] + 2 * np.pi
delta_theta = 2 * np.pi / 48
_, _, _, h = ax.hist2d(
    Fdir,
    Fmag,
    bins=[np.arange(0, 2 * np.pi + delta_theta, delta_theta), np.arange(0, 310, 10)],
    density=True,
    cmap="Blues",
    norm=colors.LogNorm(vmin=1e-4, vmax=5e-3),
)
plt.colorbar(h, label=r"PDF of F$_\mathregular{NI}$", shrink=0.5, pad=0.14)
ax.set_yticks([100, 200, 300])
ax.set_yticklabels(["100", "200", "300 W/m"], rotation=30, color="k")
ax.set_rlabel_position(300)
ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
ax.set_xticklabels(["E", "S", "W", "N"])
ax.grid(which="major", axis="y", color="k")
ax.set(theta_direction=-1)
mFuv = mFu + 1j * mFv
mFmag, mFdir = np.absolute(mFuv), np.angle(mFuv)
mFdir = mFdir - np.pi / 2
mFdir = mFdir + 2 * np.pi if mFdir < 0 else mFdir
ax.plot(mFdir, mFmag, 'wx');
gv.plot.png('mm4_ni_flux_pdf')

# %%
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection="polar")
Fuv = Fu[:, 0] + 1j * Fv[:, 0]
# Fuv = mFu + 1j * mFv
Fmag, Fdir = np.absolute(Fuv), np.angle(Fuv)
Fdir = Fdir - np.pi / 2
Fdir[Fdir < 0] = Fdir[Fdir < 0] + 2 * np.pi
delta_theta = 2 * np.pi / 48
_, _, _, h = ax.hist2d(
    Fdir,
    Fmag,
    bins=[np.arange(0, 2 * np.pi + delta_theta, delta_theta), np.arange(0, 820, 20)],
    density=True,
    cmap="Blues",
    norm=colors.LogNorm(vmin=1e-5, vmax=3e-3),
)
plt.colorbar(h, label=r"PDF of F$_\mathregular{NI}$", shrink=0.5, pad=0.14)
ax.set_yticks([100, 400, 800])
ax.set_yticklabels(["100 ", "400 ", "800 W/m "], rotation=30, color="k", ha='right', va='center')
ax.set_rlabel_position(230)
ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
ax.set_xticklabels(["E", "S", "W", "N"])
ax.grid(which="major", axis="y", color="k")
ax.set(theta_direction=-1)
mFuv = mFu + 1j * mFv
mFmag, mFdir = np.absolute(mFuv), np.angle(mFuv)
mFdir = mFdir - np.pi / 2
mFdir = mFdir + 2 * np.pi if mFdir < 0 else mFdir
ax.plot(mFdir, mFmag, 'wx');
gv.plot.png('mm4_ni_flux_pdf')

# %% [markdown]
# ## Plot Mean NI FLux on Map

# %%
print(np.sqrt(mFu**2+mFv**2))

# %% [markdown]
# Read NIKISNe mooring locations.

# %%
niskine_base_dir = Path("/Users/gunnar/Projects/niskin/")
niskine_moorings = xr.open_dataset(
    niskine_base_dir / "cruises/cruise1/py/niskine_mooring_locations.nc"
)

# %%
ss = gv.ocean.smith_sandwell(lon=os4.lon+[-5, 2], lat=os4.lat+[-1.5, 2.5])
fig, ax = gv.plot.quickfig(w=4.2)
cax = gv.plot.add_cax(fig, pad=0.1)
(-1 * ss).plot(cmap='ocean4jbm_r', vmin=0, vmax=4e3, ax=ax, cbar_ax=cax,
              cbar_kwargs=dict(label='depth [m]'))
cax.invert_yaxis()
ax.plot(niskine_moorings.lon_actual, niskine_moorings.lat_actual,
        color='#EC407A', marker='o', linestyle='', label='NISKINe')
ax.plot([os3.lon, os4.lon], [os3.lat, os4.lat], color='#FB8C00', linestyle='',
            marker='+', markersize=9, mew=2, label='OSNAP')
hq = ax.quiver(os4.lon[0], os4.lat[0], mFu, mFv, color='w', scale=3e2);
ax.legend()
ax.quiverkey(hq, 0.2, 0.9, 50, 'NI flux [50 W/m]', coordinates='axes',
            labelcolor='w')
ax.set(xlabel='', ylabel='', title='Mean OSNAP NI Flux');
gv.plot.png('mean_ni_flux_map')

# %% [markdown]
# ## Read Altimetry

# %%
altimetry_path = Path('/Users/gunnar/Projects/niskin/data/altimetry/')
altimetry_file = 'SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_04_2005_05_01_2018_01_18.nc'
alt = xr.open_dataset(altimetry_path.joinpath(altimetry_file))

# %%
alt['longitude'] = alt.longitude -360

# %%
alt = alt.rename({'longitude': 'lon', 'latitude': 'lat'})

# %%
alt['eke'] = 1/2 * (alt.ugosa**2 + alt.vgosa**2)

# %%
vort = alt.vgosa.differentiate(coord='lon') - alt.ugosa.differentiate(coord='lat')

# %%
vort.isel(time=10).plot()

# %%
t1=slice(os4.time_start, os4.time_stop)
altos = alt.eke.sel(lon=os4.lon, lat=os4.lat, method='nearest').sel(time=t1).squeeze()
altvort = vort.sel(lon=os4.lon, lat=os4.lat, method='nearest').sel(time=t1).squeeze()

# %%
fig, (ax0, ax) = plt.subplots(nrows=2, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)

altvort.plot(ax=ax0)

ax.plot(os4.time, np.cumsum(Fu[:, 0])*3600/1e9, color='0.2')
ax.plot(os4.time, np.cumsum(np.sum(Fu, axis=1))*3600/1e9, label='F$_\mathregular{u}$')
ax.plot(os4.time, np.cumsum(Fv[:, 0])*3600/1e9, color='0.2')
ax.plot(os4.time, np.cumsum(np.sum(Fv, axis=1))*3600/1e9, label='F$_\mathregular{v}$')
ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]',
       title='OSNAP MM4 Near-Inertial Energy Flux')
ax.legend()
gv.plot.concise_date()
# gv.plot.png('mm4_ni_flux_time_series')

# %%
F = Fu + 1j*Fv

# %%
np.mean(Fv[:,0])

# %%
np.mean(Fu[:,0])

# %%
Fmag = np.sqrt(Fu[:,0]**2+Fv[:,0]**2)

# %%
tmp = altvort.interp_like(os4.ctd.t)

# %%
plt.plot(tmp, Fmag, 'k.', alpha=0.2)

# %%
np.mean(Fmag[:,0])

# %%
ww = xr.open_dataarray('data/osnap_mm4_ni_windwork_integrated.nc')

# %%
fig, (ax0, ax) = plt.subplots(nrows=2, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True)

ww.plot(ax=ax0, color='0.2')
ax0.set(ylabel='P$_\mathregular{W}$ [kJ/m$^2$]', xlabel='',
        title='Near-Inertial Air-Sea Energy Flux')
gv.plot.axstyle(ax0)

ax.plot(os4.time, np.cumsum(Fu[:, 0])*3600/1e9, color='0.2')
ax.plot(os4.time, np.cumsum(np.sum(Fu, axis=1))*3600/1e9, label='F$_\mathregular{u}$')
ax.plot(os4.time, np.cumsum(Fv[:, 0])*3600/1e9, color='0.2')
ax.plot(os4.time, np.cumsum(np.sum(Fv, axis=1))*3600/1e9, label='F$_\mathregular{v}$')
ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]',
       title='OSNAP MM4 Near-Inertial Energy Flux')
ax.legend()
gv.plot.concise_date(ax)
gv.plot.axstyle(ax)
gv.plot.png('mm4_ni_flux_time_series_with_wind')

# %% [markdown]
# ## Read Clement Vic NI Flux Time Series

# %%
import datetime
import time as tm

# %%
vic = xr.open_dataset("data/forGunnar_ICE_near-inertial_fluxes.nc")
diff_days = (datetime.datetime(1950, 1, 1) - datetime.datetime(1970, 1, 1)).days
time_fmt = [tm.gmtime((i + diff_days) * 86400) for i in vic.time]
ymdhms = [
    datetime.datetime(
        time_fmt[i][0],
        time_fmt[i][1],
        time_fmt[i][2],
        time_fmt[i][3],
        time_fmt[i][4],
        time_fmt[i][5],
    )
    for i in np.arange(vic.time.shape[0])
]
victime = np.array([np.datetime64(ymdhmsi) for ymdhmsi in ymdhms])
vic = vic.assign_coords({"time": victime})
vic["fmag"] = np.sqrt(vic.fx_ni ** 2 + vic.fy_ni ** 2)

# %%
allFu, allFv = np.sum(Fu, axis=1),  np.sum(Fv, axis=1)

# %%
Fmag = np.sqrt(allFu**2 + allFv**2)

# %%
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True, sharex=True, sharey=True)
vic.fmag.plot(ax=ax[0])
ax[1].plot(os4.time, Fmag)

# %%
fig, ax = gv.plot.quickfig(h=3)
ax.plot(os4.time, Fmag, label='21.25°W', lw=0.5)
vic.fmag.plot(ax=ax, alpha=0.8, label='28.45°W', lw=0.5, color='#C2185B');
ax.legend()
ax.set(ylabel=r'$|\mathregular{F}_\mathregular{NI}|$ [W/m]',
       xlabel='',
       title='NI Flux Magnitude')
gv.plot.png('compare_ni_flux_magnitude_vic_mooring')
gv.plot.concise_date()

# %% [markdown]
# ## Todo
# - mostly DONE: clean up code, modularize
# - DONE: test a wider band for filtering
# - work with argo climatology to resolve strong seasonal cycle in mixed layer depth
# - DONE: wavemode fitting routine to python, the one I have doesn't work correctly
# - create and save diagnostic plots
# - try to subtract eta that is due to instrument motion
# - Wind work calcs. Variable mixed layer depth either from mooring or from Argo climatology needed.

# %% [markdown]
# **code modularization**
#
# We shouldn't be doing all the calculations within the mooring structure, that makes it less flexible. At the point where we have CM, ADCP and CTD data all together for one mooring, a different function (or class for that matter) should take over.
#
# Maybe creating a class for the flux calculation wouldn't be too bad. It could hold data like background stratification from climatology etc. so we don't have to re-compute if playing with a certain processing step.
#
# Also, try to make as many parameters as possible variable. Right now, these are:
# - type of climatology (WOCE / Argo WOCE)
# - bandwidth for filtering
# - central frequency for filtering (to be able to do tidal flux)
# - Bin sized for ADCP binning
# - Interpolation length for NaN in ADCP time series

# %%

# %% [markdown]
# **chatting with Matthew**
#
# - Is there a NI signal in knockdown/pressure?
#
# - try tidal fluxes
#
# - Not really flat bottom here. Talk to Tamara about NI fluxes near sloping boundary.
#
# - Zoltan + student at APL did NI fluxes from RAPID but maybe there was never anything coming out of it

# %% [markdown]
# **looking at Clement Vic's paper draft**
# - uses 1.07 as bandwidth parameter for filtering around $f$ and $M2$
# - sees mostly clockwise $f$ variance in spectra
# - Energy density: Tidal is higher than near-inertial by a factor of 2 to 3
# - While the near-inertial kinetic energy density is almost constant from IRW to ICM, ICE features a two- to threefold increase in ⟨ENI⟩ compared to other moorings. Reasons for this local maximum remain obscure as all moorings experienced statistically similar wind forcing (not shown).
# - No evidence for equatorward near-ienrtial flux
