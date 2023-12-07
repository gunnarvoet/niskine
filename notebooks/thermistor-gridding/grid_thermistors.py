# -*- coding: utf-8 -*-
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
# %matplotlib inline
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import pandas as pd
import gsw
from tqdm.notebook import tqdm

import gvpy as gv

import niskine

# %config InlineBackend.figure_format = 'retina'

# %reload_ext autoreload
# %autoreload 2
# %autosave 300

# %% [markdown]
# Load configuration from `config.yml` in the root directory. `io.load_config()` automatically detects the root directory and adjusts the paths.

# %%
cfg = niskine.io.load_config()

# %%
lon, lat, depth = niskine.io.mooring_location(mooring=1)

# %% [markdown]
# # Thermistor Gridding

# %% [markdown]
# ### Time Vector

# %% [markdown]
# Create a universal time vector for the pressure time series on M1. We will go with 20min spacing and interpolate all chipod and microcat pressure time series to this universal time vector.
#
# Let's only run through May 2020 as chipods and microcats start dropping out. Actually, some Microcats start giving up earlier in the year but maybe they are not as important when we have the chipod pressure time series. We may even be able to figure out the general behavior of the mooring during knockdown, come up with a mapping, and get away with only a few pressure measurements at the end of the time series.
#
# Update: If we want stratification until the end of the deployment we have to work with microcats only as the chipods all drop out around May. We now fit a second order polynomial to the knockdown characteristics in the microcat data. Note: This could also be done with the chipod pressure time series for finer resolution but there seem to be some issues with the chipod pressure that I don't want to deal with at the moment. I am adding pressure from ADCP 13481 though.

# %%
timeslice = niskine.io.mooring_start_end_time(mooring=1)
print(timeslice.start, "  --  ", timeslice.stop)

# %%
common_time = np.arange(timeslice.start, timeslice.stop, dtype="datetime64[20m]").astype(
    "datetime64[m]"
)

# %%
common_time = np.arange(timeslice.start, timeslice.stop, dtype="datetime64[20m]").astype(
    "datetime64[ns]"
)

# %% [markdown]
# ### M1 sensor configuration

# %% [markdown]
# The `.csv` file was exported from the [mooring config spreadsheet](https://docs.google.com/spreadsheets/d/1MQlw1ow0Y2pQBhNj85RbAa9ELnzdttzBYEEfIe2yoRk/edit#gid=2019776936) and sligthly cleaned up afterwards.

# %% [markdown]
# I moved all the info into one class for quick access. It has `config` with all sensors, `microcat`, `microcat_adcp`, `pressure`, 

# %%
m = niskine.strat.MooringConfig()

# %%
m.nomz_thermistor

# %% [markdown]
# There is also the convenience method `info` that allows for quickly checking a specific serial number.

# %%
m.info(615)

# %% [markdown]
# ### Microcat pressure time series

# %%
s = niskine.io.read_microcats(common_time=common_time)
# extract pressure only
sp = s.p
sp["mean_pressure"] = sp.mean(dim='time')
sp = sp.sortby("mean_pressure")

# %% [markdown]
# Plot full microcat pressure time series and thermistor locations

# %%
niskine.strat.plot_microcat_pressure_and_thermistor_depths()
gv.plot.concise_date()
niskine.io.png("m1_sbe37_pressure")

# %% [markdown]
# Plot microcats for a shorter time (except the deep one).

# %%
time = "2019-10"
niskine.strat.plot_microcat_pressure_segment(time)

# %% [markdown]
# ### ADCP pressure time series

# %% [markdown]
# 13481 was nearby (about 100m below) 12711 and measured throughout the time series. If it matches relatively well with 12711 then we can use it!

# %% [markdown]
# The nominal depth of the ADCP was 820m. We add 10dbar to pressure to make the no-knockdown vertical position of the ADCP match approximately that depth.

# %%
adcp = niskine.io.load_adcp(mooring=1, sn=13481)
adcp_pressure_offset = 10
adcp_pressure = adcp.pressure.interp(time=common_time) + adcp_pressure_offset

# %%
print(f"ADCP no-knockdown depth {-gsw.z_from_p(np.percentile(adcp_pressure, 0.5), lat):1.0f}m")

# %%
fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(7.5, 5), constrained_layout=True, sharex=True
)
adcp_pressure.plot(ax=ax[0], linewidth=0.5, )
ax[0].text(
    adcp_pressure.time[-1] + np.timedelta64(10, "D"),
    adcp_pressure[-1],
    "13481",
    va="center",
    color="C0",
)
sp.sel(sn=12711).plot(linewidth=0.5, ax=ax[0])
ax[0].text(
    sp.sel(sn=12711).where(~np.isnan(sp.sel(sn=12711)), drop=True).time[-1]
    + np.timedelta64(10, "D"),
    sp.sel(sn=12711).where(~np.isnan(sp.sel(sn=12711)), drop=True)[-1].data,
    "12711",
    va="center",
    color="C1",
)
ax[0].set(ylim=[1200, 700], title="")
(adcp.pressure.interp(time=sp.time) - sp.sel(sn=12711)).plot(ax=ax[1], linewidth=0.2, color="0.2")
for axi in ax:
    axi.set(title="", xlabel="")
    gv.plot.axstyle(axi, grid=False)
ax[1].set(ylabel="pressure diff [dbar]")
gv.plot.concise_date(ax[1])

# %% [markdown]
# ### Read and merge microcat and ADCP pressure time series

# %%
p = niskine.strat.gather_pressure_time_series(common_time=common_time)

# %%
niskine.strat.plot_pressure_and_thermistor_depths()
niskine.io.png("m1_sbe37_and_adcp_pressure")

# %% [markdown]
# ### Knockdown fits

# %% [markdown]
# Fit knockdown vs distance between microcats to make up for deep microcats dropping out later in the time series.

# %%
pp = niskine.strat.knockdown_analysis(4923, 2864)

# %%
pp1 = niskine.strat.knockdown_analysis(4923, 12711)

# %%
pp1 = niskine.strat.knockdown_analysis(4923, 13481)

# %%
pp2 = niskine.strat.knockdown_analysis(2864, 12711)

# %%
_ = niskine.strat.knockdown_analysis(2864, 13481)

# %%
pp3 = niskine.strat.knockdown_analysis(12711, 13481)

# %% [markdown]
# The results we get look okay. We can use these models for thermistor gridding for a time where we also have data which will let us compute an uncertainty in $N$.

# %%
fig, ax = gv.plot.quickfig()
xx, yy = pp1.linspace()
ax.plot(xx, yy, "k-")
xx, yy = pp2.linspace()
ax.plot(xx, yy, "k--")
xx, yy = pp3.linspace()
ax.plot(xx, yy, "k--")

# %% [markdown]
# Maybe we want a second set of fits that is just actual pressure vs distance to next sensor. Those fits will be easier to use in the next section where we synthetically extend the microcat pressure time series that drop out early. I implemented them in `niskine.strat.fit_knockdown_absolute` and `niskine.strat.knockdown_analysis_absolute`.

# %% [markdown]
# ### Extend short pressure time series

# %% [markdown]
# Using results from the fits above we can synthetically extend the pressure time series from microcats 12710 and 12711. This way we can then determine thermistor depth time series consistently for full records.

# %% [markdown]
# **SN12710**

# %%
sn1 = 2864
sn2 = 12710

ext_12710 = niskine.strat.extend_pressure_based_on_fit(sn2, sn1)

# %%
sn1 = 13481
sn2 = 12710

ext_12710b = niskine.strat.extend_pressure_based_on_fit(sn2, sn1)

# %% [markdown]
# I think I trust the re-construction from above more since the fit between 12711 and 13481 shows a large response (in sensor difference) during strong knockdown.

# %%
fig, ax = gv.plot.quickfig()
(ext_12710-ext_12710b).plot(color="0.2")
gv.plot.concise_date(ax)

# %% [markdown]
# **SN12711**

# %%
sn1 = 13481
sn2 = 12711

ext_12711 = niskine.strat.extend_pressure_based_on_fit(sn2, sn1)


# %% [markdown]
# Fill in the dataset with all pressure time series.

# %%
def fill_pressure_gap(sn, sp, ext):
    tmp = p.sel(sn=sn)
    mask = np.isnan(tmp)
    tmp[mask] = ext[mask]


# %%
fill_pressure_gap(12710, sp, ext_12710)
fill_pressure_gap(12711, sp, ext_12711)

# %%
fig, ax = gv.plot.quickfig()
p.plot(hue='sn', ax=ax)
ax.invert_yaxis()

# %%
niskine.strat.M.microcat_adcp

# %% [markdown]
# ### Spike on 7 Oct 2019?

# %% [markdown]
# There is definitely something there, could this have been the fishing line getting tangled in the mooring?

# %%
p.sel(time=slice("2019-10-05", "2019-10-11")).plot(hue="sn");


# %% [markdown]
# ### Compare measured pressure to nominal depth

# %%
def compare_nominal_and_measured_depth(sn, verbose=False):
    data = p.sel(sn=sn, time="2019")
    nominal = m.info(sn).depth
    minimum = np.absolute(gsw.z_from_p(data.min().item(), lat))
    half_percentile = np.absolute(gsw.z_from_p(np.percentile(data, 0.1), lat))
    if verbose:
        print("nominal:", nominal)
        print("minimum:", minimum)
        print("0.5 percentile:", half_percentile)
    return nominal, nominal - half_percentile


# %% [markdown]
# Comparing nominal depth with measured depth for all SBE37 (except the really deep one since it is not on the spreadsheet) shows that they are pretty spot on (within a meter).

# %%
sbe37sn = m.sn_microcat
diffs = []
for sn in sbe37sn:
    diffs.append(compare_nominal_and_measured_depth(sn))

# %%
diffs

# %% [markdown]
# The top two sensors are spot on. The ~2dbar difference at depth is also really close to the nominal position. Given the larger distances between sensors deeper on the mooring this offset should not matter. Also, the conversion from pressure to depth in `gsw.z_from_p` is based on a global zonal average stratification and thus has some uncertainty.

# %% [markdown]
# In the following we plot the contribution of pressure data from all microcats (except the deep one).

# %%
fig, ax = gv.plot.quickfig()
for sni in sbe37sn[:-1]:
    p.sel(sn=sni).plot.hist(bins='auto', density=True, alpha=0.3, ax=ax);
ax.set(title='');

# %% [markdown]
# ### Adjust sensor depths based on pressure time series

# %% [markdown]
# As laid out above we don't have to adjust here since the nominal depths of the pressure sensors seem to be pretty good.

# %% [markdown]
# ### Generate depth time series for each thermistor

# %% [markdown]
# How do we do this? For each time step we have a few pressure measurements along the line.
# Three cases:
# - 1) Thermistor shallower than shallowest pressure measurement. Here we simply subtract the distance to the pressure sensor from its depth time series.
# - 2) Thermistor between pressure sensors. Here we use the upper pressure measurement, convert to depth, and add the distance to the thermistor. The distance is scaled by knockdown or more precisely by the ratio of the distance to the next pressure sensor during rest and at time of measurement.
# - 3) Thermistor deeper than (usable) deepest pressure measurement. I say usable here because we have a microcat close to the bottom but that one drops out early and doesn't say much about knockdown because it sees only pressure variations within a few meters. We use the pressure time series from above the thermistor and calculate knockdown. We then scale the knockdown linearly with distance to the bottom (such that we reach zero knockdown at the bottom) and add it to the nominal thermistor depth.

# %%
all_z = []
for sn in m.sn_thermistor:
    all_z.append(niskine.strat.infer_thermistor_depth(sn, p))

zz = xr.concat(all_z, "sn")
zz.coords["sn"] = m.sn_thermistor
zz.coords["nomz"] = (["sn"], m.nomz_thermistor)

# %% [markdown]
# Plot one month as an example.

# %%
fig, ax = gv.plot.quickfig()
zz.sel(time="2020-07").swap_dims(sn="nomz").plot(hue="nomz", add_legend=False, color="0.3", linewidth=0.3)
ax.invert_yaxis()
gv.plot.concise_date()

# %%
m = niskine.strat.MooringConfig()

# %% [markdown]
# ### Read all thermistor time series

# %% [markdown]
# Reading the 2min downsampled versions here. They should all be on the same time grid. Don't read the thermistor near the bottom (it won't help much in the gridding process).

# %%
# path to data files
t2mindir = cfg.data.proc.thermistor2min

# %%
allt = []
for sn in m.sn_thermistor[:-1]:
    allt.append(xr.open_dataarray(t2mindir.joinpath(f"{sn:06d}.nc")))
    allt[-1]

# %% [markdown]
# All temperature time series start at the same time:

# %%
start_times = [ai.time[0].data for ai in allt]
np.diff(start_times)

# %% [markdown]
# Gather them into one data matrix (`xarray.concat` is super slow at combining the time series as they are of different lengths.

# %%
len_allt = [len(ai) for ai in allt]
ni = np.max(len_allt)
ii = np.argmax(len_allt)
mi = len(allt)
print(mi, ni)

# %%
time = allt[ii].time.copy()
temperature_matrix = np.ones([mi, ni]) * np.nan
for i, ai in enumerate(allt):
    temperature_matrix[i, 0:len(ai.data)] = ai.data

temp = xr.DataArray(temperature_matrix, dims=["nomz", "time"], coords=dict(time=(("time"), time.data), nomz=(("nomz"), m.nomz_thermistor[:-1])))
temp.coords["sn"] = (("nomz", m.sn_thermistor[:-1]))

# %%
temp.sel(time=slice("2019-06-08", "2019-06-15")).gv.tplot(yincrease=False)

# %%
temp.sel(time="2019-12").gv.tplot(yincrease=False)
# ax.set(ylim=[200, 20])

# %% [markdown]
# Plot monthly mean temperature profiles. Be careful to drop any thermistors that may drop out during the averaging month.

# %%
fig, ax = gv.plot.quickfig()
for i in range(6, 13):
    temp.sel(time=f"2019-{i:02d}").dropna(dim="nomz", how="any").mean(dim="time").plot(y="nomz", yincrease=False, marker=".", ax=ax)
for i in range(1, 10):
    temp.sel(time=f"2020-{i:02d}").dropna(dim="nomz", how="any").mean(dim="time").plot(y="nomz", yincrease=False, marker=".", ax=ax)
# ax.set(ylim=[200, 20])

# %% [markdown]
# ### Interpolate temperature to regular depth grid

# %% [markdown]
# Interpolate depth information to temperature time vector

# %%
# exclude the deep thermistor
zzt = zz.isel(sn=range(50))

tempz = zzt.interp(time=temp.time.data).swap_dims(sn="nomz")


temp.coords["zz"] = (("nomz", "time"), tempz.data)

# %%
fig, ax = gv.plot.quickfig()
temp.sel(time=slice("2019-11-07", "2019-11-09")).plot(x="time", y="zz", vmin=8)
ax.set(ylim=[400, 0])
gv.plot.concise_date()

# %% [markdown]
# Can we interpolate in one go or do we need to loop over all time steps? Possibly the latter. This takes a moment.

# %%
newz = np.arange(10, 1510, 10)

# %%
newtemp = np.ones((len(newz), len(temp.time))) * np.nan

# %%
index = 0
for g, tmp in tqdm(temp.groupby("time")):
    tmpzz = tmp.swap_dims(nomz="zz")
    if np.any(~np.isnan(tmpzz.zz)):
        newtemp[:, index] = tmpzz.dropna(dim="zz").interp(zz=newz).data
    index+=1

# %%
tempi = xr.DataArray(newtemp, dims=["newz", "time"], coords=dict(time=(("time"), time.data), newz=(("newz"), newz)))
tempi.attrs = dict(long_name='temperature', units='°C')
tempi = tempi.rename(newz="depth")
tempi.depth.attrs = dict(long_name='depth', units='m')

# %%
tempi.to_netcdf(cfg.data.gridded.temperature)

# %%
tempi = xr.open_dataarray(cfg.data.gridded.temperature)
tempi.close()

# %%
tempi.sel(time=slice("2019-11-07", "2019-11-09")).gv.tplot()

# %%
temp.sel(time="2019-11-08 00:00:00").plot(marker='.', y="nomz", yincrease=False)
tempi.sel(time="2019-11-08 00:00:00").plot(y="depth", yincrease=False)

# %%
tempi.sel(time=slice("2019-11-07", "2019-11-09")).differentiate(coord="depth").gv.tplot(cmap="RdBu_r")

# %% [markdown]
# Here is the onset of stratification in the spring - wondering if those features would show up in density as well (maybe check it out in microcat data):

# %%
delta = np.timedelta64(267, "D")
timesel = slice(np.datetime64("2019-07-07") + delta, np.datetime64("2019-07-11") + delta)
tempi.sel(time=timesel).differentiate(coord="depth").gv.tplot(cmap="RdBu_r")

# %%
delta = np.timedelta64(267, "D")
timesel = slice(np.datetime64("2019-07-07") + delta, np.datetime64("2019-07-11") + delta)
tempi.sel(time=timesel).gv.tplot()

# %% [markdown]
# ### Make sure long-term stratification is stable

# %%
tempi.mean(dim="time").plot(y="newz", yincrease=False, add_legend=False);
