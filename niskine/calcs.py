"""
NISKINe calculations
"""

from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import xarray as xr
import gvpy as gv

import niskine


def mixed_layer_vels():
    """Calculate average mixed layer velocity at mooring M1.

    Returns
    -------
    mlvel: xr.DataSet
        Dataset with mixed layer velocity components and mixed layer depth.
    """
    conf = niskine.io.load_config()
    if conf.data.ml.ml_vel.exists():
        mlvel = xr.open_dataset(conf.data.ml.ml_vel)
    else:
        mm = niskine.io.load_gridded_adcp(1)
        mld = niskine.io.load_mld()
        mldv = mld.interp_like(m1)
        mm["mld"] = (("time"), mldv.data)

        # could probably do this without a loop but I don't know how at the moment.
        mu = np.ones(mm.time.shape[0]) * np.nan
        mv = mu.copy()
        count = 0
        for g, mi in tqdm(mm.groupby("time")):
            mld_m = mi.where(mi.z < mi.mld, drop=True)
            mu[count] = mld_m.u.mean(...).item()
            mv[count] = mld_m.v.mean(...).item()
            count += 1

        mm["mu"] = (("time"), mu)
        mm["mv"] = (("time"), mv)

        mlvel = xr.Dataset(data_vars=dict(u=mm.mu, v=mm.mv, mld=mm.mld))
        mlvel.close()
        mlvel.to_netcdf(conf.data.ml.ml_vel)

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
