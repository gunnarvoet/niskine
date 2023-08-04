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
m1 = niskine.mooring.NISKINeMooring()

# %%
m1

# %%
m1.depth

# %%
N = niskine.flux.Flux(mooring=m1, bandwidth=1.05, runall=True, climatology="ARGO")

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "f9337f7b8cb73", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
os4 = niskine.mooring.OSNAPMooring(moorstr='UMM4')

# %%
os4.cm

# %% [markdown]
# Load gridded data for first few weeks that includes the deep ADCPs

# %%
ww = xr.open_dataset(cfg.data.gridded.adcp.joinpath("M1_gridded_may2019_simple_merge.nc"))
ww.close()

# %%
ww.u.plot()

# %% [markdown]
# Load deep ADCP only

# %%
cfg.data.proc.adcp.joinpath("M1_14408.nc")

# %%
aa = xr.open_dataset(cfg.data.proc.adcp.joinpath("M1_14408.nc"))
aa.close()

# %%
aa.u.dropna(dim="z", how="all").dropna(dim="time", how="all").plot()

# %% [markdown]
# Average in depth since we have only a few bins anyways.

# %%
u = aa.u.mean(dim="z")
v = aa.v.mean(dim="z")

u = u.interpolate_na(dim="time").dropna(dim="time")[1:]
v = v.interpolate_na(dim="time").dropna(dim="time")[1:]

u = u.resample(time="1H").mean()
v = v.resample(time="1H").mean()

# %%
cm = xr.Dataset(data_vars=dict(u=(("time"), u.data), v=(("time"), v.data)), coords=dict(time=(("time"), u.time.data)))

cm = cm.expand_dims(dim="nomz")

cm.coords["nomz"] = (("nomz"), np.array([2830]))

# %%
cm["zz"] = (("nomz", "time"), np.ones_like(cm.u)*2830)

# %%
cm

# %%
u.interpolate_na(dim="time").dropna(dim="time")[1:].plot()

# %%
u.plot()
v.plot()

# %%
u.time.diff(dim="time").data.astype("timedelta64[s]")

# %%
uu = u.dropna(dim="time")[1:]

# %% [markdown]
# Run NI bandpass filter

# %%
N.thigh

# %%
ni_low, ni_high = niskine.calcs.determine_ni_band(1.06, m1.lat)

# %%
m2_low, m2_high = niskine.calcs.determine_m2_band(1.06)

# %%
u_ni_bp = gv.signal.bandpassfilter(u, lowcut=1/ni_low, highcut=1/ni_high, fs=3600/3600, order=2)

# %%
u_m2_bp = gv.signal.bandpassfilter(u, lowcut=1/m2_low, highcut=1/m2_high, fs=3600/3600, order=2)

# %%
fig, ax = gv.plot.quickfig()
ax.plot(u-np.mean(u), color="0.3")
ax.plot(u_ni_bp, "r")
ax.plot(u_m2_bp, "b")

# %%
test = niskine.mooring.NISKINeMooring(add_bottom_adcp=True)

# %%
test_flux = niskine.flux.Flux(mooring=test, bandwidth=1.05, runall=True, climatology="ARGO")

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_up_one_time_step(test_flux, ti=50)

# %%
test_flux.up.sel(time=slice("2019-05-18", "2019-05-26")).sum(dim="mode").gv.tplot(cmap="RdBu_r", )

# %%
test_zero = niskine.mooring.NISKINeMooring(add_bottom_adcp=True, add_bottom_zero=True)

# %%
test_zero_flux = niskine.flux.Flux(mooring=test_zero, bandwidth=1.05, runall=True, climatology="ARGO")

# %%
# # %%watch -p /Users/gunnar/Projects/niskine/niskine/niskine
niskine.flux.plot_up_one_time_step(test_zero_flux, ti=50)

# %%
test_zero_flux.up.sel(time=slice("2019-05-18", "2019-05-26")).sum(dim="mode").gv.tplot(cmap="RdBu_r", )

# %%
ts = slice("2019-05-18", "2019-05-25")

# %%
fa = test_flux.flux.sel(time=ts)
fb = test_zero_flux.flux.sel(time=ts)

# %% [markdown]
# Compare mean flux with and without ADCP data at depth.

# %%
fas = fa.sum(dim="mode")
np.sqrt(fas.fx_ni**2 + fas.fy_ni**2).mean(dim="time")

# %%
fbs = fb.sum(dim="mode")
np.sqrt(fbs.fx_ni**2 + fbs.fy_ni**2).mean(dim="time")

# %%
193/200

# %%
fa.sum(dim="mode").mean(dim="time")

# %%
fb.sum(dim="mode").mean(dim="time")

# %%

# %%

# %%

# %%

# %%
fig, ax = gv.plot.quickfig(yi=True, h=3)
ax.plot(test_zero_flux.mooring.time, np.cumsum(test_zero_flux.Fu[:, 0])*3600/1e9, color='0.2')
ax.plot(test_zero_flux.mooring.time, np.cumsum(np.sum(test_zero_flux.Fu, axis=1))*3600/1e9, label='F$_\mathregular{u}$')
ax.plot(test_zero_flux.mooring.time, np.cumsum(test_zero_flux.Fv[:, 0])*3600/1e9, color='0.2')
ax.plot(test_zero_flux.mooring.time, np.cumsum(np.sum(test_zero_flux.Fv, axis=1))*3600/1e9, label='F$_\mathregular{v}$')
ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]',
       title='NISKINe M1 Near-Inertial Energy Flux')
ax.legend()
gv.plot.concise_date()
ax.grid()
niskine.io.png("niskine_m1_ni_low_mode_flux_all_with_zero_bottom_constraint", subdir="low-mode-fluxes")

# %%
test_zero_flux.flux.mean(dim="time")

# %%
Fu = test_zero_flux.flux.fx_ni.isel(mode=0)
Fv = test_zero_flux.flux.fy_ni.isel(mode=0)

# %%
mFu = Fu.mean(dim="time")
mFv = Fv.mean(dim="time")

# %%
import matplotlib.colors as colors

# %%
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection="polar")
Fuv = Fu + 1j * Fv
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
# gv.plot.png('mm4_ni_flux_pdf')

# %%
