# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python [conda env:niskine]
#     language: python
#     name: conda-env-niskine-py
# ---

# %% [markdown]
# #### Imports

# %%
# %matplotlib inline
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from pathlib import Path

import gvpy as gv

import niskine

# %reload_ext autoreload
# %autoreload 2

# %config InlineBackend.figure_format = 'retina'

cfg = niskine.io.load_config()

gv.plot.helvetica()

# %% [markdown]
# # NISKINe Plot Surface Time Series v3

# %%
rho0 = 1025

# %%
# load mooring locations
locs = xr.open_dataset(cfg.mooring_locations)
locs.close()

# %%
# load gridded temperature
tall = xr.open_dataarray(niskine.io.CFG.data.gridded.temperature)

# %%
xr.__version__

# %%
# subsample to daily averages
td = tall.resample(time="1D").mean()

# %% [markdown]
# Monthly temperature profiles together with monthly mode shapes.

# %%
m1 = niskine.mooring.NISKINeMooring(add_bottom_adcp=True, add_bottom_zero=True)

# %%
N = niskine.flux.Flux(mooring=m1, bandwidth=1.06, runall=False, climatology="ARGO")

# %%
N.background_gradients()
N.find_modes()

# %%
hmm = N.modes.hmodes.sel(mode=1).groupby("time.month").mean()
vmm = N.modes.vmodes.sel(mode=1).groupby("time.month").mean()

# %%
fig, axx = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 5),
                       constrained_layout=True, sharey=True)

months = gv.time.month_str()

# temperature
ax = axx[0]
gv.plot.cycle_cmap(n=12, cmap="plasma", ax=ax)
tm = td.sel(depth=slice(0, 1300)).groupby("time.month").mean()
for g, tmi in tm.groupby("month"):
    tmi.plot(
        y="depth", yincrease=False, add_legend=False, ax=ax, color="w", linewidth=2.5,
    )
    tmi.plot(y="depth", yincrease=False, add_legend=False, ax=ax)
colors = [plt.get_cmap("plasma")(1.0 * i / 12) for i in range(12)]
for (g, ti), col, month in zip(tm.groupby("month"), colors, months):
    xi = ~np.isnan(ti)
    x = ti.where(xi, drop=True)[0].item()
    ax.annotate(
        month,
        (x, -20),
        ha="left",
        va="center",
        color=col,
        rotation=75,
        rotation_mode="anchor",
        fontsize=9,
    )

ax.set(xlabel="$\Theta$ [°C]", title=None)

# modes
ax = axx[1]
gv.plot.cycle_cmap(n=12, cmap="plasma", ax=ax)
for g, tmi in hmm.groupby("month"):
    tmi.plot(
        y="z", yincrease=False, add_legend=False, ax=ax, color="w", linewidth=2,
    )
    tmi.plot(y="z", yincrease=False, add_legend=False, ax=ax)
for g, tmi in vmm.groupby("month"):
    tmi.plot(
        y="z", yincrease=False, add_legend=False, ax=ax, color="w", linewidth=2,
    )
    tmi.plot(y="z", yincrease=False, add_legend=False, ax=ax)

ann_opts = dict(color="0.3", bbox=dict(pad=1, facecolor="w", alpha=0.6, edgecolor="none"))
ax.annotate(r"$\eta$ modes", xy=(-0.6, 600), rotation=35, **ann_opts)
ax.annotate(r"$u$, $v$ modes", xy=(0.1, 1100), rotation=25, **ann_opts)
ax.set(xlabel="normalized mode amplitude", ylabel=None, title=None)

# mark depths where Amy plots spectra
depths = [320, 800, 1280]
xlims = ax.get_xlim()
for zi, xi in zip(depths, [0.1, -0.4, 0.7]):
    ax.hlines(zi, xlims[0], xlims[1], linestyle="--", color="0.3", linewidth=0.75, zorder=0)
    ax.annotate(f"{zi} m", (xi, zi+30), va="top", fontsize=10, **ann_opts)

for axi in axx:
    gv.plot.axstyle(axi)
    
gv.plot.subplotlabel(axx, x=0.04, y=0.95, fs=12)

# niskine.io.png("temperature_and_modes_monthly")
# niskine.io.pdf("temperature_and_modes_monthly")

# %%
# load ADCP data
adcp = niskine.io.load_gridded_adcp(mooring=1)
adcp.close()

# %%
speed, dir = gv.ocean.uv2speeddir(adcp.u, adcp.v)

# %%
ud = adcp.u.resample(time="1D").mean()

# %%
vd = adcp.v.resample(time="1D").mean()

# %%
m1eke = 0.5*rho0*(adcp.u**2 + adcp.v**2)

# %%
# load MLD vel
mldvel = xr.open_dataset(cfg.data.ml.ml_vel)
mldvel.close()
# load MLD
mld = xr.open_dataarray(cfg.data.ml.mld)
mld.close()

# %%
# calculate mixed layer KE and EKE
tlong, tshort = niskine.calcs.determine_ni_band(1.06, )
lp_period_hours = 72
ulp = niskine.calcs.lowpass_time_series(mldvel.u, lp_period_hours, fs=6)
vlp = niskine.calcs.lowpass_time_series(mldvel.v, lp_period_hours, fs=6)
mld_eke = rho0 * 0.5 * (ulp**2 + vlp **2)
mldvel["eke"] = (("time"), mld_eke)
mldvel["ke"] = (("time"), (rho0 * 0.5 * (mldvel.u**2 + mldvel.v**2)).data)

# %%
# load vorticity & EKE at M1
alt = xr.open_dataset(cfg.data.ssh_m1)
alt.close()
ssh_eke = rho0 * alt.eke

# %%
# load wind work
wind_work = xr.open_dataarray(cfg.data.wind_work.niskine_m1)
wind_work.close()
wind_work_c = xr.open_dataarray(cfg.data.wind_work.niskine_m1_cumulative)
wind_work_c.close()
wind_stress = xr.open_dataarray(cfg.data.wind_work.niskine_m1_wind_stress)
wind_stress.close()

# %%
# low-pass filter wind work
lp_period_hours = 18
wind_work_lp = wind_work.copy()
tmp = niskine.calcs.lowpass_time_series(wind_work, lp_period_hours, fs=1)
wind_work_lp.data = tmp

# %%
fig, ax = gv.plot.quickfig()
wind_work.plot(ax=ax, color="k")
wind_work_lp.plot(ax=ax, color="C6")

# %%
# load NI EKE
ni_eke = xr.open_dataarray(cfg.data.ni_eke_m1)
ni_eke.close()

# %%
mldi = mld.interp_like(ni_eke)
mld_mask = ni_eke.z < mldi
deep_mask = (ni_eke.z > 500) & (ni_eke.z < 1200)

# %%
int_all = ni_eke.where(ni_eke.z<1300).sum(dim="z") * 16 / 1e3
int_alls = int_all.rolling(time=100).mean()
int_mld = ni_eke.where(mld_mask).sum(dim="z") * 16 / 1e3
int_mlds = int_mld.rolling(time=100).mean()
int_deep = ni_eke.where(deep_mask).sum(dim="z") * 16 / 1e3
int_deeps = int_deep.rolling(time=100).mean()

# %%
out = niskine.calcs.ni_ke_by_vort_and_wind()

# %%
axd = plt.figure(layout="constrained", figsize=(10, 8)).subplot_mosaic(
    """
    A.
    B.
    C.
    D.
    E.
    FG
    H.
    """,
    height_ratios=[1, 1, 1, 1, 1, 2.5, 1],
    width_ratios=[4, 1],
    gridspec_kw={
        "wspace": 0.05,
        "hspace": 0.05,
    },
)

ax = axd["A"]
wind_stress.plot(ax=ax, linewidth=1, color="0.1")
ax.set(ylabel="[N m$^{-2}$]")
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1))
ax.annotate(
    r"wind stress $\tau$",
    xy=(0.95, 0.8),
    xycoords="axes fraction",
    backgroundcolor="w",
    ha="right",
)

ax = axd["B"]
(wind_work_lp * 1e3).plot(ax=ax, linewidth=1, color="0.1")
ax.set(ylabel="[mW m$^{-2}$]")
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2))
ax.annotate(
    r"NI wind work $\Pi_\mathrm{NI}$",
    xy=(0.95, 0.8),
    xycoords="axes fraction",
    backgroundcolor="w",
    ha="right",
)


ax = axd["C"]
wind_work_c.plot(ax=ax, linewidth=1, color="0.1")
ax.set(ylabel="[kJ m$^{-2}$]")
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2))
ax.annotate(
    r"cumulative NI wind work $\quad\int \Pi_\mathrm{NI} dt$",
    xy=(0.95, 0.1),
    xycoords="axes fraction",
    backgroundcolor="w",
    ha="right",
)

ax = axd["D"]
vort = alt.vort.sel(time=slice(wind_work.time[0].data, wind_work.time[-1].data))
ax.fill_between(vort.time, vort, 0, where=vort>0, color="C3", alpha=0.2, edgecolor=None)
ax.fill_between(vort.time, vort, 0, where=vort<0, color="C0", alpha=0.2, edgecolor=None)
vort.plot(
    ax=ax, linewidth=1, color="0.1"
)
ax.set(xlabel="", ylabel="")
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=0.1))
ax.annotate(
    r"vorticity $\zeta/f$",
    xy=(0.95, 0.1),
    xycoords="axes fraction",
    backgroundcolor="w",
    ha="right",
)

ax = axd["E"]
mldvel.eke.plot(ax=ax, color="w", linewidth=1.5)
mldvel.eke.plot(ax=ax, color="C6", linewidth=1)
ssh_eke.sel(time=slice(wind_work.time[0].data, wind_work.time[-1].data)).plot(
    ax=ax, color="w", linewidth=1.5
)
ssh_eke.sel(time=slice(wind_work.time[0].data, wind_work.time[-1].data)).plot(
    ax=ax, linewidth=1
)
ax.set(xlabel="", ylabel="[J m$^{-3}$]")
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=100))
ax.annotate(
    "EKE", xy=(0.95, 0.75), xycoords="axes fraction", backgroundcolor="w", ha="right"
)
ax.annotate(
    "ML EKE",
    xy=(0.18, 0.78),
    xycoords="axes fraction",
    ha="left",
    color="C6",
    fontsize=9,
    bbox=dict(boxstyle='square,pad=0.05', ec='none', fc='w'),
)
ax.annotate(
    "SSH EKE",
    xy=(0.18, 0.5),
    xycoords="axes fraction",
    ha="left",
    color="C0",
    fontsize=9,
    bbox=dict(boxstyle='square,pad=0.05', ec='none', fc='w'),
)

ax = axd["F"]
h = ni_eke.plot.contourf(
    ax=ax,
    extend="both",
    levels=np.arange(0.25, 4, 0.25),
    cmap="Spectral_r",
    cbar_kwargs=dict(
        aspect=30,
        shrink=0.7,
        pad=0.02,
        label="[J m$^{-3}$]",
        ticks=mpl.ticker.MaxNLocator(4),
    ),
)
gv.plot.contourf_hide_edges(h)
_ = mld.gv.tcoarsen(n=30 * 24 * 2).gv.tplot(
    ax=ax, color="w", linewidth=1.75, alpha=1
)
_ = mld.gv.tcoarsen(n=30 * 24 * 2).gv.tplot(
    ax=ax, color="k", linewidth=1, alpha=1
)
ax.invert_yaxis()
ax.set(xlabel="", ylabel="depth [m]")
ax.annotate(
    "NI KE", xy=(0.95, 0.05), xycoords="axes fraction", backgroundcolor="w", ha="right"
)

ax.set(xlim=[np.datetime64("2019-05-01"), np.datetime64("2020-10-10")])

ax = axd["G"]

ax.fill_betweenx(out["mp"].z, out["mp"]-1.96*out["mp_se"], out["mp"]+1.96*out["mp_se"], color="C3", alpha=0.2, edgecolor=None)
ax.fill_betweenx(out["mn"].z, out["mn"]-1.95*out["mn_se"], out["mn"]+1.95*out["mn_se"], color="C0", alpha=0.2, edgecolor=None)

out["mp"].plot(color="w", linewidth=2, ax=ax, y="z", yincrease=False)
out["mp"].plot(color="C3", ax=ax, y="z", yincrease=False, label=r"$\zeta/f > 0$")
out["mp_hws"].plot(color="w", ax=ax, y="z", yincrease=False, linewidth=2)
out["mp_hws"].plot(color="C3", ax=ax, y="z", yincrease=False, linestyle="--")
out["mp_lws"].plot(color="w", ax=ax, y="z", yincrease=False, linewidth=2)
out["mp_lws"].plot(color="C3", ax=ax, y="z", yincrease=False, linestyle="-.")

out["mn"].plot(color="w", linewidth=2, ax=ax, y="z", yincrease=False)
out["mn"].plot(color="C0", ax=ax, y="z", yincrease=False, label=r"$\zeta/f < 0$")
out["mn_hws"].plot(color="w", ax=ax, y="z", yincrease=False, linewidth=2)
out["mn_hws"].plot(color="C0", ax=ax, y="z", yincrease=False, linestyle="--")
out["mn_lws"].plot(color="w", ax=ax, y="z", yincrease=False, linewidth=2)
out["mn_lws"].plot(color="C0", ax=ax, y="z", yincrease=False, linestyle="-.")

ax.legend(loc="lower right")
ax.set(title="M1 $\mathrm{KE}_\mathrm{NI}$", xlabel="$\mathrm{KE}_\mathrm{NI}$ [J/m$^3$]");
ylims = axd["F"].get_ylim()
ax.set(xlim=(0, 0.8), ylabel="", ylim=ylims)
ax.yaxis.set_ticklabels([])
ax.xaxis.set_ticks([0.2, 0.4, 0.6])

ax = axd["H"]
int_alls.plot(ax=ax, linewidth=1.0, color="k")
int_mlds.plot(ax=ax, linewidth=2, color="w")
int_mlds.plot(ax=ax, linewidth=1.0, color="C6")
int_deeps.plot(ax=ax, linewidth=2, color="w")
int_deeps.plot(ax=ax, linewidth=1.0, color="C0")
ax.annotate(
    "mixed layer",
    xy=(np.datetime64("2020-03-25"), 0.8),
    ha="left",
    color="C6",
)
ax.annotate(
    "500-1200m",
    xy=(np.datetime64("2019-07-03"), 0.6),
    ha="left",
    color="C0",
)
ax.annotate(
    r"depth-integrated NI KE",
    xy=(0.95, 0.9),
    xycoords="axes fraction",
    backgroundcolor="w",
    ha="right",
)
ax.set(xlabel="", ylabel="[kJ m$^{-2}$]")
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
# somehow changing the axis position is not that simple...
# bbox = axd["H"].get_position()
# new_bbox = (bbox.x0, bbox.y0, bbox.width, bbox.height)
# axd["H"].set_position(new_bbox)

gv.plot.concise_date(axd["H"])
time_limits = axd["H"].get_xlim()
for kw, axi in axd.items():
    gv.plot.axstyle(axi)
    axi.set(title="")
#     if kw in ["A", "B", "C", "D", "E", "F"]:
    if kw in ["A", "B", "C", "D", "E", "F"]:
        gv.plot.concise_date(axd[kw])
        axi.set(xlim=time_limits)
        axi.xaxis.set_ticklabels([])

for letter in ["a", "b", "c", "d", "e", "h"]:
    gv.plot.annotate_corner(letter, axd[letter.upper()], background_circle=True, addx=-0.005, addy=-0.1)
gv.plot.annotate_corner("f", axd["F"], background_circle=True, addx=-0.005)
gv.plot.annotate_corner("g", axd["G"], background_circle=True)

plot_fig = True
if plot_fig:
    name = "surface_forcing_new2"
    niskine.io.png(name)
    niskine.io.pdf(name)

# %% [markdown]
# Plot u, v, t, mean monthly temperature profiles, and stratification

# %%
ax = plt.figure(layout="constrained", figsize=(10, 7)).subplot_mosaic(
    """
    AD
    BE
    CE
    """,
    height_ratios=[1.2, 1, 1],
    width_ratios=[2, 1],
    gridspec_kw={
        "wspace": 0.05,
        "hspace": 0.05,
    },
    sharey=False,
)

velopts = dict(vmin=-0.75, vmax=0.75, rasterized=True, yincrease=False, cmap="RdBu_r")

ud.plot(ax=ax["B"], cbar_kwargs=dict(aspect=30, shrink=0.7, pad=0.02, label="[m$\,$s$^{-1}$]", ticks=mpl.ticker.MaxNLocator(4)), **velopts)
ax["B"].annotate("eastward velocity", xy=(0.95, 0.03), xycoords="axes fraction", backgroundcolor="w", ha="right")
vd.plot(ax=ax["C"], cbar_kwargs=dict(aspect=30, shrink=0.7, pad=0.02, label="[m$\,$s$^{-1}$]", ticks=mpl.ticker.MaxNLocator(4)), **velopts)
ax["C"].annotate("northward velocity", xy=(0.95, 0.03), xycoords="axes fraction", backgroundcolor="w", ha="right")

h = td.plot.contourf(ax=ax["A"], cmap="Spectral_r", levels=np.arange(3.5, 13.5, 0.5), cbar_kwargs=dict(label="[°C]", aspect=30, shrink=0.7, pad=0.02, ticks=mpl.ticker.MaxNLocator(4)))
gv.plot.contourf_hide_edges(h)
ax["A"].set(xlim=[np.datetime64("2019-05-01"), np.datetime64("2020-10-10")])
ax["A"].set(xlabel="", ylabel="depth [m]")
ax["A"].annotate("temperature", xy=(0.95, 0.03), xycoords="axes fraction", backgroundcolor="w", ha="right")


# temperature
months = gv.time.month_str()
# ax = ax["D"]
gv.plot.cycle_cmap(n=12, cmap="plasma", ax=ax["D"])
tm = td.sel(depth=slice(0, 1300)).groupby("time.month").mean()
for g, tmi in tm.groupby("month"):
    tmi.plot(
        y="depth", yincrease=False, add_legend=False, ax=ax["D"], color="w", linewidth=2.5,
    )
    tmi.plot(y="depth", yincrease=False, add_legend=False, ax=ax["D"])
colors = [plt.get_cmap("plasma")(1.0 * i / 12) for i in range(12)]
for (g, ti), col, month in zip(tm.groupby("month"), colors, months):
    xi = ~np.isnan(ti)
    x = ti.where(xi, drop=True)[0].item()
    ax["D"].annotate(
        month,
        (x, -20),
        ha="left",
        va="center",
        color=col,
        rotation=75,
        rotation_mode="anchor",
        fontsize=9,
    )
    
# modes
gv.plot.cycle_cmap(n=12, cmap="plasma", ax=ax["E"])
for g, tmi in hmm.groupby("month"):
    tmi.plot(
        y="z", yincrease=False, add_legend=False, ax=ax["E"], color="w", linewidth=2,
    )
    tmi.plot(y="z", yincrease=False, add_legend=False, ax=ax["E"])
for g, tmi in vmm.groupby("month"):
    tmi.plot(
        y="z", yincrease=False, add_legend=False, ax=ax["E"], color="w", linewidth=2,
    )
    tmi.plot(y="z", yincrease=False, add_legend=False, ax=ax["E"])

    
# ax["E"].annotate("vertical modes", xy=(-0.6, 600), rotation=35, color="0.3")
# ax["E"].annotate("horizontal modes", xy=(0.1, 1100), rotation=25, color="0.3")
# ax["E"].set(xlabel="normalized mode amplitude", ylabel=None, title=None)

ann_opts = dict(color="0.3", bbox=dict(pad=1, facecolor="w", alpha=0.6, edgecolor="none"))
ax["E"].annotate(r"$\eta$ modes", xy=(-0.6, 600), rotation=35, **ann_opts)
ax["E"].annotate(r"$u$, $v$ modes", xy=(0.1, 1100), rotation=25, **ann_opts)
ax["E"].set(xlabel="normalized mode amplitude", ylabel=None, title=None)

# mark depths where Amy plots spectra
depths = [320, 800, 1280]
xlims = ax["E"].get_xlim()
for zi, xi in zip(depths, [0.1, -0.4, 0.7]):
    ax["E"].hlines(zi, xlims[0], xlims[1], linestyle="--", color="0.3", linewidth=0.75, zorder=0)
    ax["E"].annotate(f"{zi} m", (xi, zi+30), va="top", fontsize=10, **ann_opts)


# gv.plot.subplotlabel(ax, x=0.01, y=0.88)

for kw, axi in ax.items():
    gv.plot.axstyle(axi)
    axi.set(title="", xlabel="")
 
gv.plot.concise_date(ax["C"])

ax['A'].sharex(axd['C'])
ax['B'].sharex(axd['C'])

ax["C"].set(ylim=(1550, -30))

ax["D"].set(xlabel=r"$\Theta$ [°C]", title=None, ylabel="")
# ax["D"].yaxis.tick_right()
# ax["D"].yaxis.set_label_position('right') 
ax["D"].yaxis.set_tick_params(left=True, labelleft=True)
ax["D"].yaxis.set_ticks_position("none")
ax["D"].set(ylabel="depth [m]")
# ax["D"].spines["left"].set_visible(False)
# ax["D"].spines["right"].set_visible(True)


# ax["A"].get_shared_y_axes().remove(ax["E"])
ax["A"].set(ylim=(1500, 0))
ax["E"].set(ylim=(3000, 0))
ax["E"].yaxis.set_tick_params(left=True, labelleft=True)
ax["E"].yaxis.set_ticks_position("none")
ax["E"].set(yticks=np.arange(0, 3500, 500), ylabel="depth [m]", xlabel="normalized mode amplitude")


yoffsets = dict(A=0.03, B=0.03, C=0.03, D=0.03, E=0.07) 
for kw, axi in ax.items():
    gv.plot.annotate_corner(kw.lower(), ax=axi, background_circle=True, addy=yoffsets[kw]+0.01, addx=-0.012, fs=12)

niskine.io.png("vel_and_t_and_modes")
niskine.io.pdf("vel_and_t_and_modes")

# %%
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 5),
                       constrained_layout=True, sharex=True,
                       height_ratios=[1, 1, 1, ])

velopts = dict(vmin=-0.75, vmax=0.75, rasterized=True, yincrease=False, cmap="RdBu_r")

ud.plot(ax=ax[0], cbar_kwargs=dict(aspect=30, shrink=0.7, pad=0.02, label="[m$\,$s$^{-1}$]", ticks=mpl.ticker.MaxNLocator(4)), **velopts)
ax[0].annotate("eastward velocity", xy=(0.95, 0.03), xycoords="axes fraction", backgroundcolor="w", ha="right")
vd.plot(ax=ax[1], cbar_kwargs=dict(aspect=30, shrink=0.7, pad=0.02, label="[m$\,$s$^{-1}$]", ticks=mpl.ticker.MaxNLocator(4)), **velopts)
ax[1].annotate("northward velocity", xy=(0.95, 0.03), xycoords="axes fraction", backgroundcolor="w", ha="right")

h = td.plot.contourf(ax=ax[2], cmap="Spectral_r", levels=np.arange(3.5, 13.5, 0.5), cbar_kwargs=dict(label="[°C]", aspect=30, shrink=0.7, pad=0.02, ticks=mpl.ticker.MaxNLocator(4)))
gv.plot.contourf_hide_edges(h)
# ax[2] = mld.gv.tcoarsen(n=30*24*2).gv.tplot(ax=ax[2], color="w", linewidth=1.75, alpha=1)
# ax[2] = mld.gv.tcoarsen(n=30*24*2).gv.tplot(ax=ax[2], color="k", linewidth=1, alpha=1)
ax[2].set(xlim=[np.datetime64("2019-05-01"), np.datetime64("2020-10-10")])
ax[2].invert_yaxis()
ax[2].set(xlabel="", ylabel="depth [m]")
ax[2].annotate("temperature", xy=(0.95, 0.03), xycoords="axes fraction", backgroundcolor="w", ha="right")

gv.plot.subplotlabel(ax, x=0.01, y=0.88)

for axi in ax:
    gv.plot.axstyle(axi)
    axi.set(title="", xlabel="")
gv.plot.concise_date(ax[-1])


# %%
