# -*- coding: utf-8 -*-
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
# ### Imports

# %%
# %matplotlib inline
from pathlib import Path
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import gsw
import cmocean
import palettable
from cycler import cycler

import gvpy as gv

import niskine

# %config InlineBackend.figure_format = 'retina'

# %reload_ext autoreload
# %autoreload 2
# %autosave 300

gv.plot.helvetica()

col = palettable.wesanderson.Zissou_5.hex_colors[:5]


# %%
cfg = niskine.io.load_config()


# %% [markdown]
# Mooring location

# %%
class MLoc():
    def __init__(self):
        self.lon, self.lat, self.depth = niskine.io.mooring_location(mooring=1)
    def __repr__(self):
        return f"lon: {self.lon:1.3f}\nlat: {self.lat:1.3f}"


# %%
loc = MLoc()
loc

# %% [markdown]
# Velocity data

# %%
adcp = niskine.io.load_gridded_adcp(mooring=1)

# %% [markdown]
# Select a time series of $u$ for a specific depth.

# %%
u = adcp.u.sel(z=300, method="nearest")
u.gv.plot()

# %% [markdown]
# # Develop xr accessor

# %% [markdown]
# It would be great if we were able to run this on time series with NaN's or uneven sampling intervals.
# Lomb-Scargle periodograms can deal with randon sampling times.
# We would need to create an x vector that is simply time in seconds from the first sample.
# The downside of Lomb-Scargle is that (at least the scipy version) it does not take complex numbered time series as input and I thus don't know how to calculate a rotary spectrum this way.
#
# For now we can
# - interpolate over small gaps
# - exclude NaN's in the beginning and end
#
# and use the regular `gv.signal.psd` which is based on `scipy.signal.welch`.
# Leave Lomb-Scargle for later as a special case.

# %%
zz = 600
u = adcp.u.sel(z=zz, method="nearest")
v = adcp.v.sel(z=zz, method="nearest")

# interpolate gaps and remove nan's at beginning and end
u = u.interpolate_na(dim="time")
v = v.interpolate_na(dim="time")
mask = ~np.isnan(u) & ~np.isnan(v)
u = u[mask]
v = v[mask]

# construct a complex time series
Z = u + 1j * v
Z.attrs["lon"] = loc.lon
Z.attrs["lat"] = loc.lat

# %%
ax = Z.gv.plot_spectrum(N=2.6e-3, nwind=3)
# ax.vlines(1/8, 10, 1000)
# ax.vlines(1/4.5, 10, 1000)
ax.set(xlim=[5e-2, 10], ylim=[1e-1, 3e3]);

# %% [markdown]
# # Develop gappy rotary

# %%
# %%watch -p /Users/gunnar/Projects/python/gvpy/gvpy/

gv.signal.gappy_rotary(Z.data, nfft=1024, fs=sampling_period, maxgap=100)

# %%
# fig, ax = gv.plot.quickfig()
# ax.plot(sp.signal.windows.hann(nfft))

# %%
ZZ = Z.copy()

# %%
fig, ax = gv.plot.quickfig()
ax.plot(np.real(xdat[:nfft-1]))
ax.plot(sp.signal.detrend(np.real(xdat[:nfft] * wind), type="linear"))

# %%
x = np.real(ZZ)
y = np.imag(ZZ)

y[[0, 1, 2, 100, 101, 102, 103, 104, 105, -2, -1]] = np.nan

maxgap=2

fs = 600

nfft = 1024*16

# x = np.real(Z)
# y = np.imag(Z)

nfft = np.floor(nfft)
if nfft / 2 != np.floor(nfft / 2):
    nfft=nfft-1
nfft = nfft.astype("int")
nfft_half = (nfft / 2).astype("int")

# save the input vector
xx = np.array([x, y])

# make sure real and imag are stored as rows (note: gappy_rotary.m works in columns)
m, n = xx.shape
if m > n:
    xx = xx.transpose()

# make x the way that we find good/bad data....
x = np.sum(xx, axis=0)

# trim bad data off the front and back....
start, stop = np.flatnonzero(~np.isnan(x))[[0, -1]]
x = x[start:stop]
xx = xx[:, start:stop]

xorig = x.copy()
good = np.flatnonzero(~np.isnan(x))
t = np.arange(x.size)
gapx = xx[:, good]
gapt = t[good]
# interpolate across the gaps....
xx = interp1d(gapt, gapx, axis=1)(t)

# now find biggaps
dt = np.diff(gapt)
bad = np.flatnonzero(dt > maxgap)
bad = np.append(bad, gapt.size - 1)
goodstart = 0

# f = np.nan
# fftd=[];

fnom = np.linspace(fs/nfft, fs/2, nfft_half)

wind = sp.signal.windows.hann(nfft)
W1 = 2 / (wind @ wind)

Gxx = np.zeros(nfft)
Gyy = np.zeros(nfft)
Cxy = np.zeros(nfft)
Qxy = np.zeros(nfft)
Gxy = np.zeros(nfft)
lencount = 0

count = 0
for n in range(bad.size):
    print(n, goodstart)
    goodint = np.arange(goodstart, gapt[bad[n]], 1)
    ng = goodint.size
    if ng > nfft:
        lencount = lencount + ng
        xdat = xx[0, goodint] + 1j * xx[1, goodint]
        repeats = (np.fix(2 * xdat.size / nfft)).astype("int")
        if len(xdat) == nfft:
            repeats = 1
        X = np.fft.fft(sp.signal.detrend(np.real(xdat[:nfft]), type="linear") * wind)
        Y = np.fft.fft(sp.signal.detrend(np.imag(xdat[:nfft]), type="linear") * wind)
        Z = np.fft.fft(
            sp.signal.detrend(
                np.real(xdat[:nfft]) + 1j * np.imag(xdat[:nfft]), type="linear"
            )
            * wind
        )
        Gxx = Gxx + X * X.conjugate()
        Gyy = Gyy + Y * Y.conjugate()
        Gxy = Gxy + Z * Z.conjugate()
        Cxy = Cxy + X.real * Y.real + X.imag * Y.imag
        Qxy = Qxy + X.real * Y.imag - X.imag * Y.real
        count = count + 1

        if repeats - 1:
            step = np.fix((len(xdat) - nfft) / (repeats - 1))
            for m in np.arange(step, (len(xdat) - nfft).astype("int"), step):
                mi = m.astype("int")
                X = np.fft.fft(
                    sp.signal.detrend(xdat[mi : mi + nfft].real, type="linear") * wind
                )
                Y = np.fft.fft(
                    sp.signal.detrend(xdat[mi : mi + nfft].imag, type="linear") * wind
                )
                Z = np.fft.fft(
                    sp.signal.detrend(
                        xdat[mi : mi + nfft].real + 1j * xdat[mi : mi + nfft].imag,
                        type="linear",
                    )
                    * wind
                )
                Gxx = Gxx + X * X.conjugate()
                Gyy = Gyy + Y * Y.conjugate()
                Gxy = Gxy + Z * Z.conjugate()
                Cxy = Cxy + X.real * Y.real + X.imag * Y.imag
                Qxy = Qxy + X.real * Y.imag - X.imag * Y.real
                count = count + 1
    goodstart = gapt[bad[n]] # I think this is nested too far down in gappy_rotary.m
#         goodstart = gapt(min(bad(n)+1,length(gapt)));

# get the cw and acw components....
Gxx=W1*Gxx/count/fs
Gyy=W1*Gyy/count/fs
Gxy=W1*Gxy/count/fs
Cxy=2*W1*Cxy/count/fs
Qxy=2*W1*Qxy/count/fs

Gxx[:nfft_half] = Gxx[1:nfft_half+1]
Gyy[:nfft_half] = Gyy[1:nfft_half+1]
Gxy[:nfft_half] = Gxy[1:nfft_half+1]
Cxy[:nfft_half] = Cxy[1:nfft_half+1]
Qxy[:nfft_half] = Qxy[1:nfft_half+1]

f = np.linspace(fs / nfft, fs / 2, nfft_half)
    
# set to nan if no data
if np.sum(Gxx == 0) == nfft:
    Gxx[:] = np.nan
if np.sum(Gyy == 0) == nfft:
    Gyy[:] = np.nan
if np.sum(Gxx == 0) == nfft or np.sum(Gyy == 0) == nfft:
    Gxy[:] = np.nan
    Cxy[:] = np.nan
    Qxy[:] = np.nan
    
Gxx = Gxx[:nfft_half]
Gyy = Gyy[:nfft_half]
Gxy = np.sqrt(Cxy[:nfft_half]**2 + Qxy[:nfft_half]**2)
CW = 0.5 * (Gxx[:nfft_half] + Gyy[:nfft_half] + Qxy[:nfft_half])
CCW = 0.5 * (Gxx[:nfft_half] + Gyy[:nfft_half] - Qxy[:nfft_half])
n = 2 * lencount / nfft

# %%
fig, ax = gv.plot.quickfig()
ax.plot(f, CW)
ax.plot(tf, tCW)
ax.set(xscale="log", yscale="log")

# %%
zz = 600
u = adcp.u.sel(z=zz, method="nearest")
v = adcp.v.sel(z=zz, method="nearest")

# interpolate gaps and remove nan's at beginning and end
u = u.interpolate_na(dim="time")
v = v.interpolate_na(dim="time")
mask = ~np.isnan(u) & ~np.isnan(v)
u = u[mask]
v = v[mask]

# construct a complex time series
Z = u + 1j * v
Z.attrs["lon"] = loc.lon
Z.attrs["lat"] = loc.lat
ZZ = Z.copy()

# %%
zz = 300
u = adcp.u.sel(z=zz, method="nearest")
v = adcp.v.sel(z=zz, method="nearest")
ZZ = u + 1j * v

# %%
tf, tCW, tCCW = gv.signal.gappy_rotary(ZZ, nfft=1024*10, fs=24*3600/600, maxgap=20)

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True)
# ax.plot(tf, tCW+tCCW, color="0.5")
ax.plot(tf, tCW, linewidth=0.75, color="C3", label="CW")
ax.plot(tf, tCCW, linewidth=0.75, color="C0", label="CCW")
E = gv.gm81.calc_E_omg(N=1e-3, lat=loc.lat)
ax.plot(E.omega * 3600 * 24 / (2 * np.pi), E.KE/24/3600*2*np.pi, color='w', linewidth=2)
ax.plot(E.omega * 3600 * 24 / (2 * np.pi), E.KE/24/3600*2*np.pi, label="GM KE", color='C4')
ax.set(xscale="log", yscale="log")
ax.vlines(24/12.4, 1e-8, 1e-0, linewidth=0.5, color="0.5")
ax.grid(which="major")
ax.grid(which="minor", alpha=0.2)
ax.legend(frameon=False)
ax.set(xlim=[1e-1, 10], ylim=[1e-6, 1e-1])

# %% [markdown]
# Generate a test signal and run the gappy psd on this.

# %%
x = np.linspace(0, 100, 5000)
rng = np.random.default_rng()
y = np.sin(x * 2 * 2* np.pi) + np.sin(x * 5 * 2 * np.pi) + 2 * rng.standard_normal(len(x))
y2 = y.copy()
y2[1000:1100] = np.nan
y2[3000:3300] = np.nan

# %%
fig, ax = gv.plot.quickfig()
ax.plot(x, y2)

# %%
f, cw, ccw, gxx, gyy, gxy, n = gv.signal.gappy_rotary(y, nfft=1024*2, fs=1, maxgap=20)
# f, cw, ccw = gv.signal.gappy_rotary(y+1j*y, nfft=1024, fs=1, maxgap=2)
# f2, cw2, ccw2 = gv.signal.gappy_rotary(y2+1j*y2, nfft=1024, fs=1, maxgap=2)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5),
                       constrained_layout=True)
# ax.plot(tf, tCW+tCCW, color="0.5")
ax.plot(f, gxx, linewidth=0.75, color="C0", label="CW no gaps")
# ax.plot(f2, cw2, linewidth=0.75, color="C6", label="CW gaps")
ax.set(xscale="log", yscale="log")
ax.grid(which="major")
ax.grid(which="minor", alpha=0.2)
ax.legend(frameon=False)
# ax.set(xlim=[1e-1, 10], ylim=[1e-6, 1e-1])

# %%
n

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# %connect_info

# %%

# %% [markdown]
# # Velocity Frequency Spectra

# %% [markdown]
# Compare to GM levels. See [notes on Jody's Matlab toolbox](http://jklymak.github.io/GarrettMunkMatlab/) and Joern Callies' [python implementation](https://github.com/gunnarvoet/GM81).

# %%
sampling_period_td = adcp.time.diff('time').median().data.astype('timedelta64[s]')
sampling_period = sampling_period_td.astype(np.float64)
sampling_frequency = 1/sampling_period

# %%
sampling_period = u.gv.sampling_period

# %%
g = Z.data
Pcw, Pccw, Ptot, omega = gv.signal.psd(
    g, sampling_period, window="hann", ffttype="t", tser_window=g.size / 2
)

# %%
Ptot

# %%
test_omg, test_Pcw = sp.signal.welch(
    g,
    1 / sampling_period,
    window="hann",
    nperseg=g.size / 2,
    scaling="density",
    return_onesided=False,
)

# %%
f_cpd = gv.ocean.inertial_frequency(loc.lat)/(2*np.pi) * 3600 * 24
print(f'f = {f_cpd:.3} cpd')

# %%
Nrange_rads = np.array([1e-4, 3e-3])
Nrange_cps = Nrange_rads / 2 / np.pi
Nrange_cpd = Nrange_rads / 2 / np.pi * 3600 * 24

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), constrained_layout=True)
freqs = np.array(
    [24 / (14 * 24), 24 / 12.4, 2 * 24 / 12.4, 4 * 24 / 12.4, f_cpd, 2 * f_cpd, 1]
)
freq_labels = ["fortnightly", "M2", "2M2", "4M2", " \nf", " \n2f", "K1"]
for freq in freqs:
    ax.vlines(freq, 1e-3, 1e4, color="C0", alpha=0.5, linestyle="-", linewidth=0.75)
# ax.hlines(10**2, Nrange_cpd[0], Nrange_cpd[1], color="C0", alpha=0.8, linestyle='--')
# Spectrum
ax.plot(omega * (3600 * 24) / (2 * np.pi), Ptot * 2 * np.pi, linewidth=1, color="0.2")
# ax.plot(omega * (3600 * 24) / (2 * np.pi), np.real(Ptot), linewidth=1, color="0.2")
# GM
ax.plot(omg * 3600 * 24 / (2 * np.pi), K_omg * 2 * np.pi, label="KE", color='C3')
# ax.plot(omg * 3600 * 24 / (2 * np.pi), K_omg, label="KE", color='C6')

ax.plot([1e-1, 1], [5e2, 5e0], color="C6")

ax.set(xscale="log", yscale="log", xlim=(2.1e-2, 1e2), ylim=(1e-3, 1e5))
ax = gv.plot.axstyle(ax, ticks="in", grid=True, spine_offset=10)
gv.plot.gridstyle(ax, which="both")
gv.plot.tickstyle(ax, which="both", direction="in")
# ax2 = ax.twinx()
ax2 = ax.secondary_xaxis(location="bottom")
ax2 = gv.plot.axstyle(ax2, ticks="in", grid=False, spine_offset=30)
ax2.xaxis.set_ticks([])
ax2.xaxis.set_ticklabels([])
ax2.minorticks_off()
ax2.xaxis.set_ticks(freqs)
ax2.xaxis.set_ticklabels(freq_labels)
ax.set(ylabel='power spectral density [m$^2$/s$^2$/cps]')
ax.set_xlabel('frequency [cpd]', labelpad=35)
# gv.plot.png(f'psd_{mc.attrs["mooring"]}_{mc.attrs["SN"]}')

# %% [markdown]
# Okay, looks like there is an offset of 2π between the Welch and psd output. We do divide by 2π in the psd function to *normalize by radial wavenumber/frequency*. I guess then it makes sense to compare this with the GM frequency spectrum **not** multiplied by 2π. The match is also better.

# %% [markdown]
# I don't understand why we normalize by 2π in the psd code. I may agree that for GM things are still in wavenumber space and therefore want to be multiplied by 2π. Maybe that's just it, by dividing by 2π we go into wavenumber space, or at least scaling?

# %% [markdown]
# Note: Using `welch` with the `spectrum` scaling results in very wrong energy levels (not density anymore):

# %%
fig, ax = gv.plot.quickfig()
test_omg, test_Pcw = sp.signal.welch(
    g,
    1 / sampling_period,
    window="hann",
    nperseg=g.size / 2,
    scaling="density",
)
ax.plot(test_omg*24*3600, test_Pcw, linewidth=1, color="0.2")
test_omg2, test_Pcw2 = sp.signal.welch(
    g,
    1 / sampling_period,
    window="hann",
    nperseg=g.size / 4,
    scaling="spectrum",
)
ax.plot(test_omg2*24*3600, np.real(test_Pcw2), linewidth=1, color="r")
ax.plot(omg * 3600 * 24 / (2 * np.pi), 2*np.pi*K_omg, label="KE")
ax.set(xscale="log", yscale="log")


# %% [markdown]
# ## All deph levels

# %%
def spectra_per_variable(a, var):
    p = []
    for group, uvel in a[var].groupby('depth'):
        Pcw, Pccw, Ptot, omega = gv.signal.psd(
            uvel, sampling_period, ffttype="t", window="hanning", tser_window=uvel.size / 2
        )
        p.append(Ptot)

    Ptot = xr.DataArray(
        p,
        coords=dict(depth=a.depth.data, freq=omega * 24 * 3600 / 2 / np.pi),
        dims=["depth", "freq"],
    )
    Ptot.name = var
    Ptot = Ptot.where(Ptot.freq>0, drop=True)
    return Ptot


# %%
def spectra_all_variables(a):
    vel_vars = ['u', 'v', 'w']
    spec = []
    for var in vel_vars:
        spec.append(spectra_per_variable(a, var))
    return xr.merge(spec)


# %%
def plot_spectra_vs_depth(spec, ax):
    np.log10(spec).plot(
        robust=False,
        cmap="RdGy_r",
        cbar_kwargs=dict(
            shrink=0.7,
            label="power spectral density [m$^2$/s$^2$ / cps]",
        ),
        ax=ax,
    )
    ax.set(xscale="log")
    ax.invert_yaxis()
    ax = gv.plot.axstyle(ax, ticks="in", grid=False, spine_offset=10)
    # gv.plot.gridstyle(ax, which="both")
    gv.plot.tickstyle(ax, which="both", direction="in")
    # ax2 = ax.twinx()
    ax2 = ax.secondary_xaxis(location="bottom")
    ax2 = gv.plot.axstyle(ax2, ticks="in", grid=False, spine_offset=30)
    ax2.xaxis.set_ticks([])
    ax2.xaxis.set_ticklabels([])
    ax2.minorticks_off()
    ax2.xaxis.set_ticks(freqs)
    ax2.xaxis.set_ticklabels(freq_labels)
    ax.set(
        xlim=(1e-2, 12),
    )
    ax.set_xlabel("frequency [cpd]", labelpad=-12)
    gv.plot.annotate_corner(spec.name, ax=ax, )


# %%
def plot_spectra_all_depths(spec, ax):
    freqs = np.array(
        [24 / (14 * 24), 24 / 12.4, 2 * 24 / 12.4, 4 * 24 / 12.4, f_cpd, 2 * f_cpd, 1]
    )
    freq_labels = ["fortnightly", "M2", "2M2", "4M2", " \nf", " \n2f", "K1"]
    for freq in freqs:
        ax.vlines(freq, 1e-1, 1e1, color="C0", alpha=1, linestyle="-", linewidth=0.75)
    for depthi, spectrum in spec.groupby("depth"):
        ax.plot(
            spectrum.freq,
            spectrum.data * 2 * np.pi,
            linewidth=1,
            color="0.2",
            alpha=0.2,
        )
    # Plot GM kinetic energy frequency spectrum
    if spec.name != 'w':
        ax.plot(
            omg * 3600 * 24 / (2 * np.pi),
            K_omg * 2 * np.pi,
            label="KE",
            color="0.5",
            linestyle="--",
        )

    ax.set(xscale="log", yscale="log", xlim=(2.1e-2, 4e1), ylim=(1e-2, 5e4))
    ax = gv.plot.axstyle(ax, ticks="in", grid=True, spine_offset=10)
    gv.plot.gridstyle(ax, which="both")
    gv.plot.tickstyle(ax, which="both", direction="in")
    ax.set_ylabel("power spectral density [m$^2$/s$^2$/cps]")
    ax.set_xlabel("frequency [cpd]", labelpad=35)
    # Secondary x axis with some frequency labels
    ax2 = ax.secondary_xaxis(location="bottom")
    ax2 = gv.plot.axstyle(ax2, ticks="in", grid=False, spine_offset=30)
    ax2.xaxis.set_ticks([])
    ax2.xaxis.set_ticklabels([])
    ax2.minorticks_off()
    ax2.xaxis.set_ticks(freqs)
    ax2.xaxis.set_ticklabels(freq_labels)
    
    gv.plot.annotate_corner(spec.name, ax=ax, )


# %%
def plot_mooring_spectra(mooring):
    a = blt.adcp.quick_load_adcp(mooring=mooring)

    spec = spectra_all_variables(a)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 12),
                           constrained_layout=True)
    for var, axi in zip(spec.data_vars, ax[:, 1]):
        plot_spectra_vs_depth(spec[var], axi)
    for var, axi in zip(spec.data_vars, ax[:, 0]):
        plot_spectra_all_depths(spec[var], axi)
    fig.suptitle(a.mooring, fontsize=14, fontweight='bold')
    savename = f'velocity_spectra_{mooring}'
    blt.io.png(savename, subdir='velocity_spectra')


# %%
plot_mooring_spectra('mavs1')

# %%
plot_mooring_spectra('mavs2')

# %%
plot_mooring_spectra('mp2')

# %%
plot_mooring_spectra('mp1')

# %%

# %%

# %%

# %%

# %%

# %%

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 5), constrained_layout=True)

np.log10(Ptot).plot(
    robust=False,
    cmap="RdGy_r",
    cbar_kwargs=dict(shrink=0.7,
    label="power spectral density [m$^2$/s$^2$ / cps]",)
)
ax.set(xscale="log")
ax.invert_yaxis()
ax = gv.plot.axstyle(ax, ticks="in", grid=False, spine_offset=10)
# gv.plot.gridstyle(ax, which="both")
gv.plot.tickstyle(ax, which="both", direction="in")
# ax2 = ax.twinx()
ax2 = ax.secondary_xaxis(location="bottom")
ax2 = gv.plot.axstyle(ax2, ticks="in", grid=False, spine_offset=30)
ax2.xaxis.set_ticks([])
ax2.xaxis.set_ticklabels([])
ax2.minorticks_off()
ax2.xaxis.set_ticks(freqs)
ax2.xaxis.set_ticklabels(freq_labels)
ax.set(
    xlim=(1e-2, 12),
)
ax.set_xlabel("frequency [cpd]", labelpad=-12)

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), constrained_layout=True)
freqs = np.array(
    [24 / (14 * 24), 24 / 12.4, 2 * 24 / 12.4, 4 * 24 / 12.4, f_cpd, 2 * f_cpd, 1]
)
freq_labels = ["fortnightly", "M2", "2M2", "4M2", " \nf", " \n2f", "K1"]
for freq in freqs:
    ax.vlines(freq, 1e-1, 1e1, color="C0", alpha=1, linestyle="-", linewidth=0.75)
for depthi, spec in Ptot.groupby("depth"):
    ax.plot(
        spec.freq,
        spec.data * 2 * np.pi,
        linewidth=1,
        color="0.2",
        alpha=0.2,
    )
# Plot GM kinetic energy frequency spectrum
ax.plot(
    omg * 3600 * 24 / (2 * np.pi),
    K_omg * 2 * np.pi,
    label="KE",
    color="0.1",
    linestyle="--",
)

ax.set(xscale="log", yscale="log", xlim=(2.1e-2, 4e1), ylim=(1e-1, 5e4))
ax = gv.plot.axstyle(ax, ticks="in", grid=True, spine_offset=10)
gv.plot.gridstyle(ax, which="both")
gv.plot.tickstyle(ax, which="both", direction="in")
ax.set_ylabel("power spectral density [m$^2$/s$^2$/cps]")
ax.set_xlabel("frequency [cpd]", labelpad=35)
# Secondary x axis with some frequency labels
ax2 = ax.secondary_xaxis(location="bottom")
ax2 = gv.plot.axstyle(ax2, ticks="in", grid=False, spine_offset=30)
ax2.xaxis.set_ticks([])
ax2.xaxis.set_ticklabels([])
ax2.minorticks_off()
ax2.xaxis.set_ticks(freqs)
ax2.xaxis.set_ticklabels(freq_labels);

# %%
