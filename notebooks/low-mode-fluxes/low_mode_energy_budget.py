# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
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

# %% [markdown]
# # NISKINe Low-Mode Energy Budget

# %% [markdown]
# Here are Sam Kelly's suggestions:

# %% [markdown]
# Big picture:
#
# - Low modes (mode 1 for sure, maybe 2 and 3 as well?) are insensitive to mixed layer depth and zeta. The wind blows and they go south due to beta refraction.
# - High modes are sensitive to the mixed layer and transition layer depths. They are generated much more strongly when the ML is shallow and TL is thin. Once generated zeta refraction causes them to propagate into anti-cyclones. The speed at which they do this, I think, depends on the strength and horizontal scales of the eddy field.
#
# Demonstrating (1) and (2) would be an improvement over Alford et al. 2012 and would tie in nicely to the theme of NISKINe vs. Ocean Storms. AKA, why beta is important sometimes and zeta is important other times. 
#
# Potential analyses
#
# (1) You could compute mode-1 wind work, energy, and energy flux. You may have to "boost" wind work by a factor to account for ERA5 systematic underestimates of wind stress (maybe Thilo could give you a correction?). Mode-1 wind work should just be Pi_1 = tau_0 * u_1 * phi_1(z=0)
#
# - Divide energy by wind work to get a characteristic decay timescale, T. (This is like assuming dissipation is due to a linear drag: D = r * 2 * KE)
# - Divide energy flux by energy to get a  characteristic horizontal group speed. 
# - Use the shallow water dispersion relation to get the dominant frequency and horizontal wavenumber \[ you can use cg = sqrt(1 - f^2/omega^2) * c_n and cp = omega / k = c_n / sqrt(1 - f^2/omega^2), where c_n is the eigenspeed\]
# - Check how long after a storm it takes beta refraction to reach the observed wavenumber, where k = beta * t, maybe this will be comparable to  the decay timescale? You could also compare the observed frequency to f at a distance cg * T north.  These comparisons should hopefully make beta refraction seem plausible.
# - Try to interpret the decay timescale:
#     - Option 1: generation is uniform everywhere and decay is due to dissipation (seems unlikely). If mode-1 decays through topographic scattering, the decay rate should be similar to the upward flux of high-mode NI energy at the bottom.
#     - Option 2: generation extends some distance north of the mooring and the decay time is the time it takes the waves to clear the mooring as they propagate south. Maybe the two moorings can help you flesh this out. You can predict an increase in southward flux at the southward mooring that is proportional to the wind work between the two moorings.   
#
# (2) I have less well-formed ideas about this part. It would be really cool to plot downward high-mode energy flux at each depth. It you get a profile like figure 5 (except for downward energy flux), that would show a pretty big flux convergence over the upper 600 m during anti-cyclonic vorticity, which would be very cool.
#
# - Alford et al. 2012 has several methods of computing downward energy flux. Qi et al. (1995; JPO) also computed this quantity for Ocean Storms.
# - The flux divergence at depth might be consistent with finescale dissipation or chipod dissipation. 
# - The energy flux at the surface should correspond to the high mode generation, minus mixed-layer dissipation. You could estimate high mode generation by projecting the mixed layer (and transition layer?) profile of velocity onto vertical modes. Then you could divide the total wind work into the modal contributions. High mode work will probably be quite different for shallow vs deep mixed layers.
# - If you could somehow determine the vertical group speed for each frequency and vertical wavenumber, you could infer the horizontal wavenumber (from the dispersion relation) and compare that to eddy size. Leif or Bill also has an equation that relates the horizontal wavneumber to vorticity and time. This equation could determine whether beta or zeta refraction is more plausible for the higher modes. 

# %%
240e3/0.5/24/3600

# %%
0.6*600

# %% [markdown]
# ---

# %%
_, lat, _ = niskine.io.mooring_location(mooring=1)

# %% [markdown]
# Run the low-mode flux calculation so we have all the parameters.

# %%
m1 = niskine.mooring.NISKINeMooring(add_bottom_adcp=True, add_bottom_zero=True)

# %%
N = niskine.flux.Flux(mooring=m1, bandwidth=1.06, runall=True, climatology="ARGO")

# %% [markdown]
# What is $N$ at 800 m depth?

# %%
np.sqrt(N.N2s.mean(dim="time").sel(z=800, method="nearest").item())

# %% [markdown]
# ---

# %% [markdown]
# Calculate the beta plane approximation as $\beta = 2 \Omega \cos(\phi_0) / r_a$ where $\Omega$ is the angular rotation rate of the earth, $\phi_0$ is the central latitude, and $r_a$ the earth's radius.

# %%
beta = gv.ocean.beta(lat)
print(beta)

# %%
f = gv.ocean.inertial_frequency(lat)
print(f)

# %% [markdown]
# Following Gill (1984) the distance $y$ to the turning latitude is
# $$
# y \approx \frac{\omega_n - f}{\beta}
# $$

# %% [markdown]
# Calculate group speed from mode 1 eigenspeed

# %%
cn = N.modes.eigenspeed.sel(mode=1).mean(dim="time").item()
print(f"mode 1 eigenspeed: {cn:1.3f} m/s")

# %%
# group speed estimate from energy analysis (see fit below)
cg = 0.5
cg_low = 0.4
cg_high = 0.6

# %% [markdown]
# Pick an $\omega_n$ that solves the equation
# $$
# c_g = \sqrt{1 - \frac{f^2}{\omega_n^2}} c_n \quad .
# $$
# $\omega_n = 1.055f$ appears to be pretty close.

# %%
om_n = 1.055*f
np.sqrt(om_n**2-f**2)/om_n*cn/cg

# %%
np.sqrt(1 - f**2 / om_n**2) *cn/cg

# %%
om_n_low = 1.035*f
np.sqrt(om_n_low**2-f**2)/om_n*cn/cg_low

# %%
om_n_high = 1.078*f
np.sqrt(om_n_high**2-f**2)/om_n*cn/cg_high

# %% [markdown]
# Calculate distance to turning latitude.

# %%
dist_beta = (om_n-f) / beta / 1e3
print(f"distance to turning latitude: {dist_beta:1.0f} km")

# %% [markdown]
# Almost 600 km.

# %% [markdown]
# > eigenvalue in you problem has an eigenspeed of cn=sqrt(1/ev). The phase speed is  cp = omega/k = cn / sqrt(1-omega^2/f^2) and group speed is cg = sqrt(1-omega^2/f^2) * cn 

# %%
cp = cn / (np.sqrt(np.absolute(1 - om_n**2/f**2)))

# %%
# cp

# %%
cp = (om_n / np.sqrt(om_n**2 - f**2)) * cn 

# %%
# cp

# %% [markdown]
# What is the horizontal wavenumber of the mode 1 wave? The dispersion relation is
# $$\mathbf{k}^2 = \frac{\omega_n^2 -f^2}{c_n^2}$$
# I am slightly confused because D'Asaro et al. (1995) show the dispersion relation as
# $$\omega_n -f = \frac{c_n^2}{2f} \alpha^2$$
# where $\alpha=k^2 + l^2$ the horizontal wavenumber components. However, they refer to $c_n$ as phase velocity so maybe that's what is the difference here, not expressing the dispersion relation in terms of the eigenspeed? Let's see, their group speed is
# $$c_g = c_n^2 \frac{\alpha}{f}$$
# which is simply $\partial \omega / \partial k$ of the term above. Yes, they also have $\sigma_n$ in their equations which is their eigenspeed I think.

# %%
k = np.sqrt((om_n**2-f**2)/cn**2)
print(f"{k:1.1e}")

# %% [markdown]
# Not sure if I am missing a factor of $2\pi$ here? Usually $k = 2 \pi / \lambda$ and thus $\lambda = 2 \pi / k$ so 240 km may be the right answer. If this lateral scale is reflective of the scale of the storm we are probably not too far off though.

# %%
1/k/1e3

# %%
(2 * np.pi)/k/1e3

# %% [markdown]
# What is the difference between $f$ and $\mathrm{M}_2$ frequencies and at what period would this show up?

# %%
f = gv.ocean.inertial_frequency(lat)
omega_sd = (2*np.pi)/(12.4*3600)

# %%
2*np.pi/f/3600

# %%
((2 * np.pi) / (omega_sd - f)) / 3600 / 24

# %%
((2 * np.pi) / (omega_sd - om_n)) / 3600 / 24

# %%
((2 * np.pi) / (omega_sd + f)) / 3600 / 24

# %%
((2 * np.pi) / (omega_sd + om_n)) / 3600 / 24

# %% [markdown]
# About 4.5 days for pure f and 8 for the mode-1 frequency of $1.055 f$.

# %% [markdown]
# ---

# %% [markdown]
# Calculate mode 1 KE.

# %%
rho = 1025

# %%
ke = 0.5* rho * (N.up.sel(mode=1)**2 + N.vp.sel(mode=1)**2)

# %%
ke_i = ke.integrate(coord="z")

# %%
ax = (ke_i/1e3).gv.tplot()
ax.set(ylabel="mode 1 KE [kJ/m$^2$]")

# %% [markdown]
# Load the mode 1 wind work.

# %%
m1ww = xr.open_dataarray(cfg.data.wind_work.niskine_m1)
m1ww = m1ww.interp_like(ke_i)

# %%
m1ww.gv.tplot()

# %% [markdown]
# Divide energy by wind-work. 

# %%
decay_timescale = (ke_i / m1ww)

# %%
fig, ax = gv.plot.quickfig()
(decay_timescale/3600/24).plot.hist(ax=ax, bins=np.arange(-50, 51, 1));
ax.set(title=None, xlabel="decay timescale [days]")


# %% [markdown]
# How do we deal with negative wind work? Let's filter for positive wind-work for now. Regress KE on wind-work as in [Vic et al, 2020](zotero://select/items/@vicetal21).

# %%
def decay_timescale(ww, ke, ax):

    # fit all positive data
    mask = ww > 0
    x = ww[mask]
    y = ke[mask]
    mask = np.isfinite(x) & np.isfinite(y)
    A = np.vstack([x[mask], np.ones(len(x[mask]))]).T
    slope, offset = np.linalg.lstsq(A, y[mask], rcond=None)[0]

    print(f"decay time scale: {slope/3600/24:1.1f} days")
    print(f"offset {offset:1.0f} J/m$^2$")

    # calculate average mode 1 KE in mode 1 wind-work bins
    mask = ww > 0
    ke_ww = xr.DataArray(ke[mask].data/1e3, coords=dict(ww=ww[mask].data*1e3), dims=["ww"])
    ke_ww_binned = ke_ww.groupby_bins("ww", bins=np.arange(0, 1.1, 1e-1),
                                      labels=np.arange(0.05, 1.05, 0.1)).mean()

    # Linearly fit the bin-averaged mode 1 KE. Note that we have to multiply the slope by $10^6$ because the data is in mW/m$^2$ vs kJ/m$^2$. The vertical offset has to be multiplied by $10^3$ because of the kilo-scaling of the y-axis.
    slope2, c2 = ke_ww_binned.polyfit(dim="ww_bins", deg=1).polyfit_coefficients.data * [1e6, 1e3]
    print(f"decay time scale: {slope2/3600/24:1.1f} days")

    h, xedge, yedge, img = ax.hist2d(
        ww[mask] * 1e3,
        ke[mask] / 1e3,
        bins=30,
        range=[[0, 2], [0, 1.200]],
        norm=mpl.colors.LogNorm(vmin=1, vmax=1e3),
        density=False,
        cmap="inferno",
        rasterized=True,
    )
    plt.colorbar(img, shrink=0.7, label="pdf [counts]")
    ax.set(xlabel="mode 1 wind-work [mW/m$^2$]", ylabel="mode 1 KE [kJ/m$^2$]")
    xx = np.arange(0, 2.1, 0.1)

    ax.plot(xx, (slope * xx / 1e3 + offset) / 1e3, color="w", linewidth=3, linestyle="-")
    ax.plot(xx, (slope * xx / 1e3 + offset) / 1e3, color="C0", linewidth=2, linestyle="-")

    ke_ww_binned.plot(
        ax=ax, mfc="C6", mec="w", marker="o", linewidth=3, linestyle="",
    )

    xx = np.arange(0, 1.1, 0.1)
    ax.plot(xx, (slope2 * xx / 1e3 + c2) / 1e3, color="w", linewidth=3, linestyle="-")
    ax.plot(xx, (slope2 * xx / 1e3 + c2) / 1e3, color="C6", linewidth=2, linestyle="-")
    ax.set(xlabel="mode 1 wind-work [mW/m$^2$]", ylabel="mode 1 KE [kJ/m$^2$]", title="NISKINE mode 1 decay time scale")


# %%
fig, ax = gv.plot.quickfig(fgs=(5, 4))
decay_timescale(m1ww, ke_i, ax)

# %% [markdown]
# What is the decay time scale with data from winter only? Still about 5 days but maybe a day and a half a day faster.

# %%
ts = slice("2020-01", "2020-03")
fig, ax = gv.plot.quickfig(fgs=(5, 4), fs=12)
decay_timescale(m1ww.sel(time=ts), ke_i.sel(time=ts), ax)


# %% [markdown]
# ---

# %% [markdown]
# Divide energy flux by energy to get a  characteristic horizontal group speed. What are the units here? This is W/m divided by J/m$^2$ which is m/s (W=J/s).

# %%
def group_speed(ke, flux, ax):
    cu = N.flux.fx_ni.sel(mode=1) / ke_i
    cv = N.flux.fy_ni.sel(mode=1) / ke_i

    c_amp = np.sqrt(cu**2 + cv**2)

    flux_amp = np.sqrt(N.flux.fx_ni.sel(mode=1) ** 2 + N.flux.fy_ni.sel(mode=1) ** 2)

    x = flux_amp
    y = ke_i

    #     mask = np.isfinite(x) & np.isfinite(y)
    #     A = np.vstack([x[mask], np.ones(len(x[mask]))]).T

    #     slope, offset = np.linalg.lstsq(A, y[mask], rcond=None)[0]

    # We can also calculate average mode 1 flux in mode 1 KE bins.

    flux_amp_ke = xr.DataArray(flux_amp.data, coords=dict(ke=ke_i.data), dims=["ke"])

    flux_ke_binned = flux_amp_ke.groupby_bins(
        "ke", bins=np.arange(0, 850, 50), labels=np.arange(25, 825, 50)
    ).mean()

    # Linearly fit the bin-averaged mode 1 flux.

    slope_flux, offset_flux = flux_ke_binned.polyfit(
        dim="ke_bins", deg=1
    ).polyfit_coefficients.data

    print(f"{slope_flux}")

    slope_flux_low, offset_flux_low = (
        flux_ke_binned.where(flux_ke_binned.ke_bins < 200)
        .polyfit(dim="ke_bins", deg=1)
        .polyfit_coefficients.data
    )

    print(f"{slope_flux_low}")

    h, xedge, yedge, img = ax.hist2d(
        #     np.absolute(m1ww * 1e3),
        #     ke_i / 1e3,
        ke_i,
        flux_amp,
        bins=50,
        range=[[0, 1300], [0, 800]],
        norm=mpl.colors.LogNorm(vmin=1, vmax=1e3),
        density=False,
        cmap="inferno",
        rasterized=True,
    )
    plt.colorbar(img, shrink=0.7, label="pdf [counts]")

    flux_ke_binned.plot(
        ax=ax,
        mfc="C6",
        mec="w",
        marker="o",
        linewidth=3,
        linestyle="",
    )

    xx = np.arange(0, 900, 100)
    ax.plot(xx, (slope_flux * xx + offset_flux), color="w", linewidth=3, linestyle="-")
    ax.plot(xx, (slope_flux * xx + offset_flux), color="C6", linewidth=2, linestyle="-")

    xx = np.arange(0, 210, 10)
    ax.plot(
        xx,
        (slope_flux_low * xx + offset_flux_low),
        color="w",
        linewidth=3,
        linestyle="-",
    )
    ax.plot(
        xx,
        (slope_flux_low * xx + offset_flux_low),
        color="C0",
        linewidth=2,
        linestyle="-",
    )

    ax.set(
        ylabel="mode 1 energy flux [W/m$^2$]",
        xlabel="mode 1 KE [J/m$^2$]",
        title="NISKINE mode 1 group speed",
    )


# %%
fig, ax = gv.plot.quickfig(fgs=(5, 4), fs=12)
group_speed(ke_i, N.flux, ax)

# %%
# fig, ax = gv.plot.quickfig(fgs=(5, 4), fs=12, grid=True)
# c_amp.plot.hist(bins=np.arange(0, 4.1, 0.1), rasterized=True);
# ax.set(title=None, xlabel="${c_\mathrm{g}}_\mathrm{h}$ [m/s]", ylabel="pdf [count]");
# niskine.io.pdf("mode1_cgh", subdir="low-mode-fluxes")
# niskine.io.png("mode1_cgh", subdir="low-mode-fluxes")

# %% [markdown]
# ---

# %% [markdown]
# Plot both fits together.

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                       constrained_layout=True, gridspec_kw={'wspace': 0.1})

group_speed(ke_i, N.flux, ax[0])
decay_timescale(m1ww, ke_i, ax[1])
for axi in ax:
    gv.plot.axstyle(axi, fontsize=12)

# gv.plot.subplotlabel(ax, fs=13, x=0.03, y=0.98)
gv.plot.annotate_corner("g", ax[0], addy=0.04, fs=12)
gv.plot.annotate_corner("h", ax[1], addy=0.04, fs=12)


niskine.io.png("mode1_decay_time_scale_and_group_speed")
niskine.io.pdf("mode1_decay_time_scale_and_group_speed")


# %% [markdown]
# ---

# %% [markdown]
# Determine how much forcing region is needed to drive the observed low-mode energy flux.
#
# 74 W/m divided by 0.12 mW/m^2 $\approx$ 600 km.

# %%
upstream_forcing_km = 74/0.12
print(f"upstream forcing region: {upstream_forcing_km:1.0f} km")

# %% [markdown]
# ---

# %% [markdown]
# Integrate turbulent dissipation (average $2\times 10^{-9}$ W/kg in the upper few hundred meters in Kunze et al. (2023)).

# %% [markdown]
# Units are W/kg * kg/m^3 * m  = W/m^2

# %%
H = 500
eps = 2e-9
rho0 = 1025

# %%
dissipation_integrated = H*eps*rho0 * 1e3

# %%
print(f"depth-integrated turbulent dissipation: {dissipation_integrated:1.3f} mW/m^2")
