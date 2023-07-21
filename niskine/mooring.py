"""
Create mooring data structures that can be used for flux calculations.
"""

from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
import netCDF4
import warnings
from pathlib import Path
import gsw
import gvpy as gv
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import niskine


class Mooring(ABC):

    bandwidth = 1.05

    def __repr__(self):
        return f"data structure for mooring {self.name}"

    @abstractmethod
    def add_location_data(self):
        pass

    @abstractmethod
    def common_time_vector(self):
        pass

    def shorten(self, timespan):
        """
        Return a shortened version.

        Parameters
        ----------
        timespan : datestr or slice(datestr, datestr)
            Time span to cut out. Can be a single string like `'2015-02'`
            or a time span given by two strings as e.g.
            `slice('2015-02-01', '2015-02-10')`

        Returns
        -------
        Shortmoooring
        """
        return Shortmooring(self, timespan)


class Shortmooring:
    """
    Cut out a time segment from a Mooring object with all data necessary for
    flux calculations.

    Parameters
    ----------
    mooring : osnap.Mooring
        Mooring object
    timespan : datestr or slice(datestr, datestr)
        Time span to cut out. Can be a single string like `'2015-02'`
        or a time span given by two strings as e.g.
        `slice('2015-02-01', '2015-02-10')`
    """

    def __init__(self, mooring, timespan):
        self.name = mooring.name
        self.adcp = mooring.adcp.sel(time=timespan)
        if hasattr(mooring, "cm"):
            self.cm = mooring.cm.sel(time=timespan)
        if hasattr(mooring, "ctd"):
            self.ctd = mooring.ctd.sel(time=timespan)
        self.lon = mooring.lon
        self.lat = mooring.lat
        self.depth = mooring.depth
        self.time = self.adcp.time.data


class NISKINeMooring(Mooring):
    """Gather data from NISKINe mooring M1 into a Dataset.

    Data interpolated in time and depth are in ctd, adcp, cm.
    """

    def __init__(self):
        self.cfg = niskine.io.load_config()
        self.mooring = 1
        self.name = f"NISKINe M{self.mooring}"
        self.add_location_data()
        self.load_adcp()
        self.common_time_vector()
        self.load_thermistor_data()

    def add_location_data(self):
        lon, lat, bottom_depth = niskine.io.mooring_location(mooring=self.mooring)
        self.lon = lon
        self.lat = lat
        self.depth = bottom_depth
        self.bottom_depth = bottom_depth

    def load_adcp(self):
        # Read gridded ADCP data and subsample to one hour.
        self.adcp = (
            niskine.io.load_gridded_adcp(mooring=self.mooring)
            .resample(time="1H")
            .mean()
        )
        self.adcp = self.adcp.transpose("z", "time", "adcp")

    def load_thermistor_data(self):
        # t = xr.open_dataarray(self.cfg.data.gridded.temperature_10m_nc)
        t = xr.open_dataarray(self.cfg.data.gridded.temperature)
        t = t.interp(time=self.time)
        # Interpolate to less depth levels. This should probably be done right
        # away when interpolating.

        # Load thermistor data from near the bottom and merge with the rest
        tdeep = xr.open_dataarray(
            "/Users/gunnar/Projects/niskine/data/NISKINe/Moorings/NISKINE19/M1/RBRSolo/proc/072174_20201010_0625.nc"
        )
        tdeep = tdeep.interp_like(t)
        tdeep = tdeep.expand_dims("depth")
        # add depth of thermistor, determined via mean pressure record of CTD below.
        tdeep.coords["depth"] = np.array([2580])
        tt = xr.concat([t, tdeep], dim="depth").sortby("depth")

        p = gsw.p_from_z(-tt.depth.data, self.lat)
        p = np.tile(p, (len(t.time), 1)).transpose()
        # We will read the temperature data into `ctd` to match what we are
        # doing with moored CTDs in the OSNAP dataset.
        ctd = xr.Dataset(data_vars=dict(tt=tt.rename(depth="z")))

        # Calculate potential temperature. For now we use a constant salinity
        # but could probably use salinity interpolated from the SBE37s later
        # on.
        ctd["th"] = (("z", "time"), gsw.pt0_from_t(35, tt, p).data)
        # self.ctd["CT"] = (
        #     ["time", "z"],
        #     gsw.CT_from_t(self.ctd.SA, self.ctd.t, self.ctd.p).data,
        # )
        # self.ctd["th"] = (
        #     ["time", "nomz"],
        #     gsw.pt_from_CT(self.ctd.SA, self.ctd.CT).data,
        # )

        # Add deep SBE37
        sbe37 = xr.open_dataset(
            "/Users/gunnar/Projects/niskine/data/NISKINe/Moorings/NISKINE19/M1/SBE37/proc/SN12712/sbe37_12712_niskine.nc"
        )
        sbe37 = sbe37.where(sbe37.p > 2850, drop=True)
        sbe37 = sbe37.interp_like(ctd.th)
        sbe = xr.Dataset(
            data_vars=dict(tt=sbe37.t, th=gsw.pt_from_CT(sbe37.SA, sbe37.CT))
        )
        sbe = sbe.expand_dims("z")
        # add depth of CTD, determined via mean pressure record.
        sbe.coords["z"] = np.array([2845])
        self.ctd = xr.concat([ctd, sbe], dim="z").sortby("z")

        # pick specific depths
        znew = np.arange(50, 750, 50)
        znew = np.append(znew, [830, 930, 1030, 1180, 1330, 2580, 2845])
        self.ctd = self.ctd.sel(z=znew)

        # add coordinate nomz to match OSNAP mooring
        self.ctd.coords["nomz"] = self.ctd.z.copy()

        # add depth matrix
        zz = np.tile(self.ctd.z, (len(self.ctd.time), 1))
        self.ctd["zz"] = (("z", "time"), zz.transpose())
        print("TODO: Use non-interpolated temperature")

    def common_time_vector(self):
        self.time = self.adcp.time.data


class OSNAPMooring(Mooring):
    """Convert all data files from an OSNAP mooring into a Dataset.

    Individual data files are listed in variables
    ctdlist, adcplist, cmlist.

    Data interpolated in time and depth are in ctd, adcp, cm.

    Parameters
    ----------
    moorstr : str
        Mooring name, for example 'mm4', used to locate mooring files in the
        raw data structure.
    """

    def __init__(self, moorstr):
        # I/O
        cfg = niskine.io.load_config()
        self._osnap_data = cfg.data.input.osnap
        self._moorstr = moorstr
        self.name = f"OSNAP {moorstr}"
        self.allfiles = self.get_osnap_mooring_files()
        self.adcpfiles = [fi for fi in self.allfiles if "ADCP" in fi.name]
        self.cmfiles = [fi for fi in self.allfiles if "CM" in fi.name]
        self.mctdfiles = [fi for fi in self.allfiles if "MCTD" in fi.name]

        # Read ADCP files
        self.adcplist = []
        for i, fi in enumerate(self.adcpfiles):
            self.adcplist.append(self.read_adcp_nc_file(fi))

        # Read Current Meter files
        self.cmlist = []
        for i, fi in enumerate(self.cmfiles):
            self.cmlist.append(self.read_cm_nc_file(fi))

        # Read CTD files
        self.ctdlist = []
        for i, fi in enumerate(self.mctdfiles):
            self.ctdlist.append(self.read_ctd_nc_file(fi))

        self.common_time_vector()
        self.add_location_data()
        self.interpolate_time_ctd()
        self.ctd_calcs()
        self.interpolate_time_cm()
        self.cm_calcs()
        self.interpolate_time_adcp()

        # self.bandpass_ctd()

    def get_osnap_mooring_files(self):
        """
        Create a list of all files for this mooring.

        Returns
        -------
        f : list
            List with paths to data files.

        Note
        ----
        This should work with the original data structure
        as downloaded from the Duke server:
        https://research.repository.duke.edu/concern/datasets/9593tv14n?locale=en

        """
        # we know the directories are unique
        pattern = "*{}*".format(self._moorstr.upper())
        d = next(self._osnap_data.glob(pattern))
        # now look for all files
        f = list(d.glob(pattern))
        return f

    def read_cm_nc_file(self, filename):
        """Read current meter data from netcdf file.

        Parameters
        ----------
        filename : netcdf
            Current meter data from OSNAP data repository.

        Returns
        -------
        ds : xarray.Dataset
            Current meter data.
        """
        ds = xr.open_dataset(filename)
        changenames = dict(
            LATITUDE="lat",
            LONGITUDE="lon",
            TIME="time",
            PRES="p",
            VCUR="v",
            UCUR="u",
            WCUR="w",
            ECUR="error",
            DEPTH="z",
        )
        for k, v in changenames.items():
            if k in ds:
                ds = ds.rename({k: v})
        return ds

    def read_adcp_nc_file(self, filename):
        """Read ADCP data from netcdf file.

        Parameters
        ----------
        filename : netcdf
            ADCP data from OSNAP data repository.

        Returns
        -------
        ds : xarray.Dataset
            ADCP data.
        """
        ds = xr.open_dataset(filename, drop_variables="BINDEPTH")
        changenames = dict(
            LATITUDE="lat",
            LONGITUDE="lon",
            TIME="time",
            INSTRDEPTH="instdep",
            BINDEPTH="bindep",
            PRES="p",
            VCUR="v",
            UCUR="u",
            WCUR="w",
            ECUR="error",
            DEPTH="z",
        )
        for k, v in changenames.items():
            if k in ds:
                ds = ds.rename({k: v})
        # need to read bin depths separately using netcdf4 library directly
        ds = ds.rename_dims({"BINDEPTH": "z"})
        nc_fid = netCDF4.Dataset(filename, "r")
        bindepths = nc_fid.variables["BINDEPTH"][:]
        ds.coords["bindepth"] = (["time", "z"], bindepths)
        # copy attributes
        bd = nc_fid.variables["BINDEPTH"]
        for attr in bd.ncattrs():
            ds.bindepth.attrs[attr] = bd.getncattr(attr)
        nc_fid.close()
        return ds

    def read_ctd_nc_file(self, filename):
        """Read moored CTD data from netcdf file.

        Parameters
        ----------
        filename : netcdf
            Moored CTD data from OSNAP data repository.

        Returns
        -------
        ds : xarray.Dataset
            CTD data.
        """
        ds = xr.open_dataset(filename, drop_variables="BINDEPTH")
        changenames = dict(
            LATITUDE="lat",
            LONGITUDE="lon",
            TIME="time",
            PRES="p",
            TEMP="t",
            PSAL="s",
            DEPTH="z",
        )
        for k, v in changenames.items():
            if k in ds:
                ds = ds.rename({k: v})
        return ds

    def common_time_vector(self):
        """Generate a common time vector for all instruments.

        Note
        ----
        Currently hard coded to one hour intervals.
        """
        self.n = []
        self.tmin = []
        self.tmax = []
        for ai in self.adcplist:
            self.n.append(len(ai.time))
            self.tmin.append(ai.time.min().data)
            self.tmax.append(ai.time.max().data)
        for ai in self.cmlist:
            self.n.append(len(ai.time))
            self.tmin.append(ai.time.min().data)
            self.tmax.append(ai.time.max().data)
        for ai in self.ctdlist:
            self.n.append(len(ai.time))
            self.tmin.append(ai.time.min().data)
            self.tmax.append(ai.time.max().data)
        self.time_start = np.datetime64(np.min(self.tmin), "h")
        self.time_stop = np.datetime64(np.max(self.tmax), "h")
        self.time = np.arange(self.time_start, self.time_stop, dtype="datetime64[h]")

    def get_position(self):
        """Add lon/lat as class attributes."""
        self.lon = self.cmlist[0].lon.data.item()
        self.lat = self.cmlist[0].lat.data.item()

    def get_bottom_depth(self):
        ss = -1 * (
            gv.ocean.smith_sandwell(lon=np.array([self.lon]), lat=np.array([self.lat]))
            .load()
            .squeeze()
        )
        self.depth = ss.data
        self.depth = ss.data
        self.bottom_depth = ss.item()

    def add_location_data(self):
        self.get_position()
        self.get_bottom_depth()

    def interpolate_time_ctd(self):
        """Interpolate CTD data to a common time vector."""
        nctd = []
        for ci in self.ctdlist:
            # xarray (2023.6.0) complains when using non-nanosecond precision
            # in the interpolation
            nctd.append(ci.interp(time=self.time.astype('datetime64[ns]')))
        self.ctd = xr.concat(nctd, dim="z")
        self.ctd = self.ctd.rename({"z": "nomz"})
        self.ctd = self.ctd.sortby("nomz")

    def interpolate_time_cm(self):
        """Interpolate current meter data to a common time vector."""
        ncm = []
        for ci in self.cmlist:
            # xarray (2023.6.0) complains when using non-nanosecond precision
            # in the interpolation
            ncm.append(ci.interp(time=self.time.astype('datetime64[ns]')))
        self.cm = xr.concat(ncm, dim="z")
        self.cm = self.cm.rename({"z": "nomz"})
        self.cm = self.cm.sortby("nomz")

    def interpolate_time_adcp(self):
        """Interpolate ADCP data in time and depth."""
        # time interpolation
        nadcp = []
        for ci in self.adcplist:
            # xarray (2023.6.0) complains when using non-nanosecond precision
            # in the interpolation
            nadcp.append(ci.interp(time=self.time.astype('datetime64[ns]')))

        # depth interpolation
        znew = np.arange(0, 810, 10)
        ui = np.full((len(self.time), len(znew)), np.nan)
        vi = np.full((len(self.time), len(znew)), np.nan)

        allu = [ai.u for ai in nadcp]
        allv = [ai.v for ai in nadcp]
        allb = [ai.bindepth for ai in nadcp]

        U = np.concatenate(allu, axis=1)
        V = np.concatenate(allv, axis=1)
        B = np.concatenate(allb, axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, (Ui, Vi, Bi) in enumerate(zip(U, V, B)):
                ni = np.isfinite(Ui)
                if len(np.flatnonzero(ni)) > 4:
                    ui[i, :] = interp1d(
                        Bi[ni], Ui[ni], bounds_error=False, fill_value=np.nan
                    )(znew)
                ni = np.isfinite(Vi)
                if len(np.flatnonzero(ni)) > 4:
                    vi[i, :] = interp1d(
                        Bi[ni], Vi[ni], bounds_error=False, fill_value=np.nan
                    )(znew)

        # xarray (2023.6.0) complains when using non-nanosecond precision time
        # here...
        adcp = xr.Dataset(
            data_vars={"u": (["time", "z"], ui), "v": (["time", "z"], vi)},
            coords={"time": (["time"], self.time.astype('datetime64[ns]')), "z": (["z"], znew)},
        )
        self.adcp = adcp.transpose("z", "time")

    def ctd_calcs(self):
        """
        Calculate absolute salinity, conservative and potential temperature
        and depth.

        Store them in the CTD Dataset.
        """
        self.ctd["SA"] = (
            ["time", "nomz"],
            gsw.SA_from_SP(self.ctd.s, self.ctd.p, self.lon, self.lat).data,
        )
        self.ctd["CT"] = (
            ["time", "nomz"],
            gsw.CT_from_t(self.ctd.SA, self.ctd.t, self.ctd.p).data,
        )
        self.ctd["th"] = (
            ["time", "nomz"],
            gsw.pt_from_CT(self.ctd.SA, self.ctd.CT).data,
        )
        self.ctd["zz"] = (
            ["time", "nomz"],
            -1 * gsw.z_from_p(self.ctd.p, self.lat).data,
        )
        self.ctd = self.ctd.squeeze().transpose("nomz", "time")

    def cm_calcs(self):
        self.cm["zz"] = (["time", "nomz"], -1 * gsw.z_from_p(self.cm.p, self.lat).data)
        self.cm = self.cm.squeeze()
        if "nomz" in self.cm.dims:
            self.cm = self.cm.transpose("nomz", "time")

    def save_to_netcdf(self):
        """Save data from CTD / CM / ADCP in OSNAP mooring data structure to netcdf
        files.
        """
        # Get output path from config
        cfg = niskine.io.load_config()
        savedir = cfg.data.gridded.osnap
        savedir.mkdir(exist_ok=True)

        outfile_base = self._moorstr.lower()

        # Save CTD
        ctd = self.ctd.drop(["SA", "CT", "th"])
        savefile = outfile_base + "_ctd.nc"
        savepath = savedir / savefile
        ctd.to_netcdf(savepath)
        # Save CM
        savefile = outfile_base + "_cm.nc"
        savepath = savedir / savefile
        self.cm.to_netcdf(savepath)
        # Save ADCP
        savefile = outfile_base + "_adcp.nc"
        savepath = savedir / savefile
        self.adcp.to_netcdf(savepath)
