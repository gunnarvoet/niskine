"""
NISKINe calculations
"""

import time
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import gsw
import gvpy as gv

import niskine


class MixedLayerDepth:
    """Calculate mixed layer depth. We want to be able to run this quickly for
    a subset and for varying criteria.
    """

    def __init__(self):
        self.lon, self.lat, _ = niskine.io.mooring_location(mooring=1)
        self.load_temperature()
        self.calc_potential_temperature()
        self.determine_highest_temp()
        self.determine_knockdown()
        self.load_sst()
        self.load_argo_mld_climatology()

    def load_temperature(self):
        progress("load temperature data")
        temp = xr.open_dataarray(niskine.io.CFG.data.gridded.temperature)
        self.temp = temp.dropna(dim="time", how="all")

    def load_sst(self):
        """Load SST data."""
        progress("load SST data")
        sst_file = niskine.io.CFG.data.sst.joinpath("sst_m1.nc")
        self.sst = xr.open_dataset(sst_file)
        self.sst.close()
        # interpolate to temperature time series
        self.ssti = self.sst.sst.interp_like(self.ptemp)

    def load_argo_mld_climatology(self):
        """Load Argo MLD climatology."""
        progress("load Argo MLD climatology")
        argo_file = niskine.io.CFG.data.ml.mld_argo
        self.argo_mld = xr.open_dataset(argo_file)
        self.argo_mld.close()
        # fit spline to interpolate to mooring time vector
        argo_time = np.arange(
            "2019-01", "2021-01", dtype="datetime64[M]"
        ) + np.timedelta64(14, "D")
        # For `CubicSpline` to work with the `datetime64` format both original
        # and new time vector have to be in the exact same format.
        argo_time = argo_time.astype("datetime64[ns]")
        argo_data = np.tile(self.argo_mld.dt_m1.data, 2)
        pch = PchipInterpolator(argo_time, argo_data)
        self.argo_mld_i = xr.DataArray(
            data=pch(self.temp.time.data), coords=dict(time=self.temp.time.data)
        )

    def calc_potential_temperature(self):
        """Calculate potential temperature."""
        progress("calculate potential temperature")
        self.temp.coords["p"] = gsw.p_from_z(-self.temp.depth, self.lat)
        self.ptemp = gsw.pt0_from_t(35, self.temp, self.temp.p)
        self.ptemp.attrs["long_name"] = r"$\theta$"

    def pick_subset(self, timespan):
        self.subset = self.ptemp.sel(time=timespan)

    def determine_knockdown(self):
        """Determine index and depth of shallowest data point."""
        mask = ~np.isnan(self.temp)
        self.shallowest_index = mask.argmax(dim="depth")
        self.depth_shallowest = self.temp.depth.isel(depth=self.shallowest_index)

    def determine_highest_temp(self):
        """Determine index and depth of warmest temperature."""
        self.highest_temp = self.ptemp.max(dim="depth")
        self.highest_temp_index = self.ptemp.argmax(dim="depth")
        self.depth_highest_temp = self.ptemp.depth.isel(depth=self.highest_temp_index)

    def calc_mixed_layer_depth(self, tcrit):
        delta_temp = self.ptemp - self.highest_temp
        temp_in_range = delta_temp.where(delta_temp + tcrit > 0)
        mask_in_range = ~np.isnan(temp_in_range)
        ind = (
            mask_in_range.shape[0] - np.argmax(mask_in_range.data[::-1, :], axis=0) - 1
        )
        mld = xr.DataArray(
            mask_in_range.depth.isel(depth=ind),
            coords=dict(time=mask_in_range.time.data),
            dims=["time"],
        )
        return mld

    def calc_sst_mask(self, sst_criterion=1.5):
        self.mask_sst = (self.ssti - self.highest_temp) < sst_criterion

    def calc_knockdown_mask(self, knockdown_criterion=100):
        self.mask_knockdown = self.depth_shallowest < 100

    def calc_final_mld_product(self, tcrit, sst_criterion=1.5, knockdown_criterion=100):
        mld = self.calc_mixed_layer_depth(tcrit)
        self.mld = mld
        self.calc_sst_mask(sst_criterion)
        self.calc_knockdown_mask(knockdown_criterion)
        # apply sst and knockdown masks
        tmp = mld.where(self.mask_sst & self.mask_knockdown)
        # smooth with moving average
        tmpm = tmp.rolling(time=30 * 8, min_periods=30 * 2).mean()
        # blank out July/August
        tmpms = tmpm.where((tmpm["time.month"] < 7) | (tmpm["time.month"] > 8))
        # interpolate over small-ish gps
        mld_c = tmpms.interpolate_na(dim="time", max_gap=np.timedelta64(4, "D"))
        mask = np.isnan(mld_c)
        # fill rest with Argo MLD climatology
        mld_c[mask] = self.argo_mld_i[mask]
        self.mld_c = mld_c
        return mld_c

    def plot_steps_and_final_mld(self):
        mpl.rcParams["lines.linewidth"] = 1
        gv.plot.helvetica()

        fig, ax = plt.subplots(
            nrows=4, ncols=1, figsize=(10, 7), constrained_layout=True, sharex=True
        )

        # SST
        self.sst.sst.plot(color="C0", zorder=5, ax=ax[0])
        _data_array_annotate(
            ax[0], self.sst.sst, "2019-07-01", "SST", (-20, 13.5), color="C0"
        )
        ax[0].fill_between(
            self.sst.time.data,
            self.sst.sstmin.data,
            self.sst.sstmax.data,
            color="C0",
            alpha=0.3,
            zorder=4,
        )
        self.highest_temp.plot(ax=ax[0], color="0.7")
        self.highest_temp.where(self.ssti - self.highest_temp < 1.5).plot(
            ax=ax[0], color="k"
        )
        _data_array_annotate(
            ax[0],
            self.highest_temp,
            "2019-10-01",
            "mooring highest temperature",
            (-40, 7.8),
            color="k",
        )
        ht = ax[0].set(ylabel="temperature [°C]")

        # knockdown
        self.depth_shallowest.plot(ax=ax[1], yincrease=False, color="0.5")
        self.depth_shallowest.where(self.depth_shallowest < 100).plot(
            ax=ax[1], yincrease=False, color="k"
        )
        _data_array_annotate(
            ax[1],
            self.depth_shallowest,
            "2019-12-01",
            "depth top buoy",
            (-45, 350),
            color="k",
        )

        # MLD with masks and Argo MLD climatology
        self.mld.plot(ax=ax[2], color="0.5", yincrease=False)
        self.mld.where(self.mask_sst & self.mask_knockdown).plot(ax=ax[2], color="k")
        _data_array_annotate(
            ax[2], self.mld, "2019-11-01", "MLD 0.2° criterion", (-45, 450), color="k"
        )
        ax[2].plot(self.temp.time, self.argo_mld_i, color="C0", linewidth=1.5)
        _data_array_annotate(
            ax[2],
            self.argo_mld_i,
            "2019-08-01",
            "Argo MLD climatology",
            (-40, 0),
            color="C0",
        )

        # merged MLD product
        self.mld_c.plot(ax=ax[3], color="k", yincrease=False)
        _data_array_annotate(
            ax[3],
            self.mld_c,
            "2019-11-01",
            "MLD",
            (-40, 320),
            color="k",
        )

        gv.plot.concise_date_all()
        for axi in ax:
            gv.plot.axstyle(axi)
            axi.set(xlabel="", title="")
        for axi in ax[[1, 2, 3]]:
            axi.set(ylim=[650, -100])
            axi.yaxis.set_major_locator(mpl.ticker.MultipleLocator(250))

        gv.plot.concise_date_all()

    def plot_argo_climatology_comparison(self):
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(6, 4), constrained_layout=True
        )
        self.mld.groupby("time.month").mean().plot(
            color="C7", linestyle="--", yincrease=False, label="NISKINe M1 MLD no qc"
        )
        self.mld.where(self.mask_knockdown & self.mask_sst).groupby(
            "time.month"
        ).mean().plot(
            color="C3", linestyle="-", yincrease=False, label="NISKINe M1 MLD"
        )
        self.mld_c.groupby("time.month").mean().plot(
            color="C0", yincrease=False, label="NISKINe M1 MLD w/ Argo"
        )
        self.argo_mld.da_m1.plot(
            color="C4", yincrease=False, label="Argo MLD Climatology (algorithm)"
        )
        self.argo_mld.dt_m1.plot(
            color="C6", yincrease=False, label="Argo MLD Climatology [threshold]"
        )
        ax.legend()
        ax.set(ylabel="MLD [m]")
        gv.plot.axstyle(ax)

    def save_results(self):
        cfg = niskine.io.CFG
        out = xr.Dataset(
            data_vars=dict(
                mld_no_qc=(("time"), self.mld.data),
                mld=(("time"), self.mld_c.data),
                sst=(("time"), self.ssti.data),
                mask_sst=(("time"), self.mask_sst.data),
                mask_knockdown=(("time"), self.mask_knockdown.data),
                argo_mld=(("time"), self.argo_mld_i.data),
            ),
            coords=dict(time=(("time"), self.mld.time.data)),
        )
        out.mld.attrs = dict(long_name='MLD', units='m')
        out.sst.attrs = dict(long_name='SST', units='°C')
        out.mask_sst.attrs = dict(long_name='SST criterion mask', units='bool')
        out.mask_knockdown.attrs = dict(long_name='knockdown criterion mask', units='bool')
        out.argo_mld.attrs = dict(long_name='Argo MLD climatology', units='m')
        out.to_netcdf(cfg.data.ml.mld_with_extras)
        out.mld.to_netcdf(cfg.data.ml.mld)


def mixed_layer_vels():
    """Calculate average mixed layer velocity at mooring M1.

    Returns
    -------
    mlvel: xr.DataSet
        Dataset with mixed layer velocity components and mixed layer depth.
    """
    cfg = niskine.io.load_config()
    mm = niskine.io.load_gridded_adcp(1)
    mld = niskine.io.load_mld()
    mldv = mld.interp_like(mm)
    mm["mld"] = (("time"), mldv.data)

    def calc_mld_vel(vel, mld):
        # calculate mixed layer average
        mld_vel_average = vel.where(vel.z < mld).mean(dim="z")
        # find velocity in shallowest bin
        mask_vel = ~np.isnan(vel)
        ui = mask_vel.argmax(dim="z")
        first_bin_vel = vel.isel(z=ui)
        # combine the two (fill with shallowest where we don't have mld average vel)
        mask_vel_average = np.isnan(mld_vel_average)
        mld_vel_average[mask_vel_average] = first_bin_vel[mask_vel_average]
        return mld_vel_average

    mld_u = calc_mld_vel(mm.u, mm.mld)
    mld_v = calc_mld_vel(mm.v, mm.mld)

    mm["mu"] = (("time"), mld_u.data)
    mm["mv"] = (("time"), mld_v.data)

    mlvel = xr.Dataset(data_vars=dict(u=mm.mu, v=mm.mv, mld=mm.mld))
    mlvel.close()
    mlvel.to_netcdf(cfg.data.ml.ml_vel)

    return mlvel


def bandpass_time_series(data, tlow, thigh, minlen=120, fs=1):
    """
    Band-pass filter time series data.

    Filters chunks of data if there are any NaNs in between.

    Parameters
    ----------
    data : array_like
        Time series data
    tlow : float
        cutoff period low
    thigh : float
        cutoff period high

    Returns
    -------
    lpt : array_like
        Band-passed data
    """
    gg = np.flatnonzero(np.isfinite(data))
    blocks = _consec_blocks(gg, combine_gap=0)
    lpt = np.full_like(data, np.nan)
    for bi in blocks:
        bb = data[bi[0] : bi[1]]
        if bi[1] - bi[0] > minlen:
            bpdata = gv.signal.bandpassfilter(
                bb, lowcut=1 / tlow, highcut=1 / thigh, fs=fs, order=2
            )
            lpt[bi[0] : bi[1]] = bpdata
    return lpt


class NIWindWork(ABC):

    s_w = 0.3
    bandwidth = 1.05

    @abstractmethod
    def load_mooring_location(self):
        pass

    @abstractmethod
    def load_surface_velocities(self):
        pass

    def determine_start_and_end_times(self):
        self.start_time = self.vel.time.min().data
        self.stop_time = self.vel.time.max().data

    def determine_ni_band(self):
        # Determine NI band limits (in hours)
        t = gv.ocean.inertial_period(lat=self.lat, verbose=False) * 24
        self.ni_band_period_short = 1 / self.bandwidth * t
        self.ni_band_period_long = self.bandwidth * t

    def load_era5(self):
        # Load ERA5 winds
        era5uvwind = niskine.io.load_wind_era5()
        # Interpolate to mooring location
        u10 = era5uvwind.u10.interp(lat=self.lat, lon=self.lon).squeeze()
        v10 = era5uvwind.v10.interp(lat=self.lat, lon=self.lon).squeeze()
        u10.attrs["units"] = r"m s$^{-1}$"
        u10.attrs["long_name"] = "10 m wind"
        v10.attrs["units"] = r"m s$^{-1}$"
        v10.attrs["long_name"] = "10 m wind"
        # Cut down to time of mooring
        u10 = u10.where(
            (u10.time > self.start_time) & (u10.time < self.stop_time), drop=True
        )
        v10 = v10.where(
            (v10.time > self.start_time) & (v10.time < self.stop_time), drop=True
        )
        self.u10 = u10
        self.v10 = v10

    def calculate_relative_wind_speed(self, apply_current_feedback_correction=True):
        # First interpolate ocean velocities to time vector of ERA5 winds
        self.u_ocean = self.vel.u.interp_like(self.u10)
        self.v_ocean = self.vel.v.interp_like(self.u10)
        # Now that we have ocean velocity and wind at the same time stamps
        # let's make sure that the sampling period is one hour:
        assert (
            gv.time.convert_units(
                self.u_ocean.time.diff(dim="time").median().data, unit="s"
            )
            == 3600
        )
        # Calculate the wind speed relative to the ocean surface.
        if apply_current_feedback_correction:
            # Apply correction factor as done in Vic et al (2021)
            self.u_wind_relative = self.u10 - (1 - self.s_w) * self.u_ocean
            self.v_wind_relative = self.v10 - (1 - self.s_w) * self.v_ocean
        else:
            # No current feedback correction factor
            self.u_wind_relative = self.u10 - self.u_ocean
            self.v_wind_relative = self.v10 - self.v_ocean

    def calculate_wind_stress(self):
        # Calculate wind stress from the wind relative to the ocean surface.
        self.taux, self.tauy = gv.ocean.wind_stress(
            self.u_wind_relative, self.v_wind_relative
        )

    def bandpass_windstress(self):
        taux_ni = bandpass_time_series(
            self.taux, self.ni_band_period_long, self.ni_band_period_short, fs=1
        )
        self.taux_ni = self.taux.copy()
        self.taux_ni.data = taux_ni
        tauy_ni = bandpass_time_series(
            self.tauy, self.ni_band_period_long, self.ni_band_period_short, fs=1
        )
        self.tauy_ni = self.tauy.copy()
        self.tauy_ni.data = tauy_ni

    def bandpass_surface_vel(self):
        u_ocean_ni = bandpass_time_series(
            self.u_ocean, self.ni_band_period_long, self.ni_band_period_short, fs=1
        )
        self.u_ocean_ni = self.u_ocean.copy()
        self.u_ocean_ni.data = u_ocean_ni
        v_ocean_ni = bandpass_time_series(
            self.v_ocean, self.ni_band_period_long, self.ni_band_period_short, fs=1
        )
        self.v_ocean_ni = self.v_ocean.copy()
        self.v_ocean_ni.data = v_ocean_ni

    def calculate_wind_work(self):
        # Run all steps prior to the wind work calculation
        self.load_mooring_location()
        self.load_surface_velocities()
        self.determine_ni_band()
        self.load_era5()
        self.calculate_relative_wind_speed(self.apply_current_feedback_correction)
        self.calculate_wind_stress()
        self.bandpass_windstress()
        self.bandpass_surface_vel()
        # Calculate NI wind work
        self.wind_work = self.taux_ni * self.u_ocean_ni + self.tauy_ni * self.v_ocean_ni
        self.wind_work.attrs = dict(
            long_name="near-inertial air-sea energy flux", units="W/m$^2$"
        )
        self.wind_work.time.attrs = dict(long_name=" ")
        # Integrate in time
        self.wind_work_int = (self.wind_work * 3.6e3 / 1e3).cumsum(
            dim="time",
        )
        self.wind_work_int.attrs = dict(
            long_name="cumulative near-inertial air-sea energy flux", units="kJ/m$^2$"
        )
        self.wind_work_int.time.attrs = dict(long_name=" ")


class NIWindWorkNiskine(NIWindWork):
    def __init__(self, apply_current_feedback_correction=True):
        self.apply_current_feedback_correction = apply_current_feedback_correction
        self.calculate_wind_work()

    def load_mooring_location(self):
        # NISKINe mooring locations
        self.lon, self.lat, dep = niskine.io.mooring_location(mooring=1)

    def load_surface_velocities(self):
        # For NISKINe M1 we use velocities averaged over the mixed layer.
        self.vel = mixed_layer_vels()
        # Interpolate over NaNs, otherwise the bandpass filter throws out a lot of data.
        self.vel = self.vel.interpolate_na(dim="time")
        self.determine_start_and_end_times()


class NIWindWorkOsnap(NIWindWork):
    def __init__(self, mooring=3, apply_current_feedback_correction=True):
        self.mooring_id = mooring
        self.apply_current_feedback_correction = apply_current_feedback_correction
        self.calculate_wind_work()

    def load_mooring_location(self):
        # NISKINe mooring locations
        m = niskine.osnap.Mooring(moorstr=f"UMM{self.mooring_id}")
        self.lon, self.lat = m.lon[0], m.lat[0]
        # keep the mooring data around for other calcs
        self.mooring_data = m

    def load_surface_velocities(self):
        m = self.mooring_data
        u = m.adcp.u.where(m.adcp.z < 200, drop=True).mean(dim="z")
        v = m.adcp.v.where(m.adcp.z < 200, drop=True).mean(dim="z")
        self.vel = xr.merge([u, v])
        # Interpolate over NaNs, otherwise the bandpass filter throws out a lot of data.
        self.vel = self.vel.interpolate_na(dim="time")
        self.determine_start_and_end_times()


def _consec_blocks(idx=None, combine_gap=0, combine_run=0):
    """
    block_idx = consec_blocks(idx,combine_gap=0, combine_run=0)

    Routine that returns the start and end indexes of the consecutive blocks
    of the index array (idx). The second argument combines consecutive blocks
    together that are separated by <= combine. This is useful when you want
    to perform some action on the n number of data points either side of a
    gap, say, and don't want that action to be effected by a neighbouring
    gap.

    From Glenn Carter, University of Hawaii
    """
    if idx.size == 0:
        return np.array([])

    # Sort the index data and remove any identical points
    idx = np.unique(idx)

    # Find the block boundaries
    didx = np.diff(idx)
    ii = np.concatenate(((didx > 1).nonzero()[0], np.atleast_1d(idx.shape[0] - 1)))

    # Create the block_idx array
    block_idx = np.zeros((ii.shape[0], 2), dtype=int)
    block_idx[0, :] = [idx[0], idx[ii[0]]]
    for c in range(1, ii.shape[0]):
        block_idx[c, 0] = idx[ii[c - 1] + 1]
        block_idx[c, 1] = idx[ii[c]]

    # Find the gap between and combine blocks that are closer together than
    # the combine_gap threshold
    gap = (block_idx[1:, 0] - block_idx[0:-1, 1]) - 1
    if np.any(gap <= combine_gap):
        count = 0
        new_block = np.zeros(block_idx.shape, dtype=int)
        new_block[0, 0] = block_idx[0, 0]
        for ido in range(block_idx.shape[0] - 1):
            if gap[ido] > combine_gap:
                new_block[count, 1] = block_idx[ido, 1]
                count += 1
                new_block[count, 0] = block_idx[ido + 1, 0]
        new_block[count, 1] = block_idx[-1, 1]
        block_idx = new_block[: count + 1, :]

    # Combine any runs that are shorter than the combine_run threshold
    runlength = block_idx[:, 1] - block_idx[:, 0]
    if np.any(runlength <= combine_run):
        count = 0
        new_block = np.zeros(block_idx.shape, dtype=int)
        for ido in range(block_idx.shape[0]):
            if runlength[ido] > combine_run:
                new_block[count, :] = block_idx[ido, :]
                count += 1
        block_idx = new_block[:count, :]

    return np.atleast_2d(block_idx)


def progress(s):
    for i in range(5):
        print(".", sep="", end="")
        time.sleep(0.1)
    print(s, end="")
    for i in range(5):
        time.sleep(0.1)
        print(".", sep="", end="")
    print("")


def _data_array_annotate(ax, da, time, text, offset, **kwargs):
    time = np.datetime64(time)
    x = time
    y = da.sel(time=time, method="nearest").item()
    ax.annotate(
        text, xy=(x, y), xytext=offset, textcoords=("offset points", "data"), **kwargs
    )
