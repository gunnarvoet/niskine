"""
Analysis of OSNAP mooring data for NISKINe.
This is the old collection of code I wrote before starting the niskine package.
"""

import numpy as np
import xarray as xr
from pathlib import Path
import netCDF4
from scipy.interpolate import interp1d
import warnings
import gsw
import gvpy as gv
import scipy
import matplotlib.pyplot as plt
import pandas as pd

import niskine


class Mooring:
    """Convert all data files from an OSNAP mooring into a Dataset.

    Individual data files are listed in variables
    ctdlist, adcplist, cmlist.

    Data interpolated in time are in ctd, adcp, cm.

    Parameters
    ----------
    moorstr : str
        Mooring name, for example 'mm4', used to locate mooring files in the
        raw data structure.
    """

    def __init__(self, moorstr):
        # I/O
        conf = niskine.io.load_config()
        self._osnap_data = conf.data.input.osnap
        self._moorstr = moorstr
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
        self.get_position()
        self.get_bottom_depth()
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
        """Add lon/lat as class attributes.
        """
        self.lon = self.cmlist[0].lon.data
        self.lat = self.cmlist[0].lat.data

    def get_bottom_depth(self):
        ss = -1 * (gv.ocean.smith_sandwell(lon=self.lon, lat=self.lat).load().squeeze())
        self.depth = ss.data

    def interpolate_time_ctd(self):
        """Interpolate CTD data to a common time vector.
        """
        nctd = []
        for ci in self.ctdlist:
            nctd.append(ci.interp(time=self.time))
        self.ctd = xr.concat(nctd, dim="z")
        self.ctd = self.ctd.rename({"z": "nomz"})
        self.ctd = self.ctd.sortby("nomz")

    def interpolate_time_cm(self):
        """Interpolate current meter data to a common time vector.
        """
        ncm = []
        for ci in self.cmlist:
            ncm.append(ci.interp(time=self.time))
        self.cm = xr.concat(ncm, dim="z")
        self.cm = self.cm.rename({"z": "nomz"})
        self.cm = self.cm.sortby("nomz")

    def interpolate_time_adcp(self):
        """Interpolate ADCP data in time and depth.
        """
        # time interpolation
        nadcp = []
        for ci in self.adcplist:
            nadcp.append(ci.interp(time=self.time))

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

        adcp = xr.Dataset(
            data_vars={"u": (["time", "z"], ui), "v": (["time", "z"], vi)},
            coords={"time": (["time"], self.time), "z": (["z"], znew)},
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
        self.ctd["th"] = (["time", "nomz"], gsw.pt_from_CT(self.ctd.SA, self.ctd.CT).data)
        self.ctd["zz"] = (["time", "nomz"], -1 * gsw.z_from_p(self.ctd.p, self.lat).data)
        self.ctd = self.ctd.squeeze().transpose("nomz", "time")

    def cm_calcs(self):
        self.cm["zz"] = (["time", "nomz"], -1 * gsw.z_from_p(self.cm.p, self.lat).data)
        self.cm = self.cm.squeeze()
        if "nomz" in self.cm.dims:
            self.cm = self.cm.transpose("nomz", "time")

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

    def save_to_netcdf(self):
        """Save data from CTD / CM / ADCP in OSNAP mooring data structure to netcdf
        files.
        """
        # Get output path from config
        conf = niskine.io.load_config()
        savedir = conf.data.gridded.osnap
        savedir.mkdir(exist_ok=True)

        outfile_base = self._moorstr.lower()

        # Save CTD
        ctd = self.ctd.drop(['SA', 'CT', 'th'])
        savefile = outfile_base + '_ctd.nc'
        savepath = savedir / savefile
        ctd.to_netcdf(savepath)
        # Save CM
        savefile = outfile_base + '_cm.nc'
        savepath = savedir / savefile
        self.cm.to_netcdf(savepath)
        # Save ADCP
        savefile = outfile_base + '_adcp.nc'
        savepath = savedir / savefile
        self.adcp.to_netcdf(savepath)


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
        self._moorstr = mooring._moorstr
        self.cm = mooring.cm.sel(time=timespan)
        self.adcp = mooring.adcp.sel(time=timespan)
        self.ctd = mooring.ctd.sel(time=timespan)
        self.lon = mooring.lon
        self.lat = mooring.lat
        self.depth = mooring.depth
        self.time = self.ctd.time.data


class Flux:
    """
    Calculate flux time series for OSNAP mooring.

    Parameters
    ----------
    mooring : osnap.Mooring
        Mooring data generated with `osnap.Mooring()`.
    bandwidth : float
        Bandwidth parameter
    type : str ['NI', 'M2'], optional
        Type of flux, determines center frequency for bandpass filter.
        Default 'NI'.
    climatology : str ['ARGO', 'WOCE'], optional
        Select type of climatology to use for calculation of modes and
        vertical displacements. Default 'ARGO' which resolves the seasonal
        cycle.
    runall : bool, optional
        Set to False to run all processing steps manually. Default True.

    """

    def __init__(
        self, mooring, bandwidth, nmodes=3, type="NI", climatology="WOCE", runall=True
    ):
        self.mooring = mooring
        self.bandwidth = bandwidth
        self.climatology = climatology
        self.nmodes = nmodes
        self.band()
        if runall:
            self.background_gradients()
            self.find_modes()
            self.bandpass()
            self.eta_modes()
            self.vel_modes()
            self.calc_pp()
            self.flux_calcs()

    def __repr__(self):
        return f"flux calculations for OSNAP mooring {self.mooring._moorstr}"

    def band(self):
        t = gv.ocean.inertial_period(lat=self.mooring.lat[0]) * 24
        self.thigh = 1 / self.bandwidth * t
        self.tlow = self.bandwidth * t

    def background_gradients(self):
        if self.climatology == "ARGO":
            self.N2s, self.Tzs = climatology_argo_woce(self.mooring)
            # interpolate to mooring time vector
            self.N2 = interpolate_seasonal_data_to_mooring(self.mooring, self.N2s)
            self.Tz = interpolate_seasonal_data_to_mooring(self.mooring, self.Tzs)
        elif self.climatology == "WOCE":
            self.N2, self.Tz = climatology_woce(self.mooring)

    def find_modes(self):
        if self.climatology == "WOCE":
            self.modes = calc_modes(self.mooring, self.N2, nmodes=self.nmodes)
        elif self.climatology == "ARGO":
            self.modes = calc_modes_seasonal(self.mooring, self.N2s, nmodes=self.nmodes)

    def bandpass(self):
        self.mooring = bandpass_ctd(self.mooring, tlow=self.tlow, thigh=self.thigh)
        self.mooring = bandpass_cm(self.mooring, tlow=self.tlow, thigh=self.thigh)
        self.mooring = bandpass_adcp(self.mooring, tlow=self.tlow, thigh=self.thigh)

    def eta_modes(self):
        self.mooring = calc_eta(self.mooring, self.Tz)
        self.beta_eta = project_eta_on_modes(
            self.mooring, self.modes, nmodes=self.nmodes
        )

    def vel_modes(self):
        self.binbpu, self.binbpv = downsample_adcp_data(self.mooring.adcp)
        self.beta_u, self.beta_v = project_vels_on_modes(
            self.mooring, self.binbpu, self.binbpv, self.modes, nmodes=self.nmodes
        )

    def calc_pp(self):
        self.pp = calculate_pressure_perturbation(
            self.mooring, self.beta_eta, self.modes, self.N2, nmodes=self.nmodes
        )

    def flux_calcs(self):
        self.up, self.vp = calculate_up_vp(
            self.mooring, self.beta_u, self.beta_v, self.modes, nmodes=self.nmodes
        )
        self.Fu, self.Fv = calculate_ni_flux(self.up, self.vp, self.pp, self.N2)
        self.flux = xr.Dataset(
            data_vars={
                "fx_ni": (
                    ["time", "mode"],
                    self.Fu,
                    {"long_name": "NI Flux U", "units": "W/m"},
                ),
                "fy_ni": (
                    ["time", "mode"],
                    self.Fv,
                    {"long_name": "NI Flux V", "units": "W/m"},
                ),
            },
            coords={
                "time": (["time"], self.mooring.time),
                "mode": (
                    ["mode"],
                    self.modes.mode,
                    {"long_name": "mode", "units": ""},
                ),
            },
        )


def near_inertial_energy_flux(mooring, bandwidth, N2, Tz):
    """Calculate near-inertial energy flux time series for one mooring.

    """
    t = gv.ocean.inertial_period(lat=mooring.lat[0]) * 24
    thigh = 1 / bandwidth * t
    tlow = bandwidth * t
    print("Filter parameters: {:1.2f}h {:1.2f}h {:1.2f}h".format(tlow, t, thigh))
    Fu, Fv = calculate_flux(mooring, N2, Tz, tlow, thigh)
    return Fu, Fv


def tidal_m2_energy_flux(mooring, bandwidth, N2, Tz):
    t = 12.4
    thigh = 1 / bandwidth * t
    tlow = bandwidth * t
    print("Filter parameters: {:1.2f}h {:1.2f}h {:1.2f}h".format(tlow, t, thigh))
    Fu, Fv = calculate_flux(mooring, N2, Tz, tlow, thigh)
    return Fu, Fv


def calculate_flux(mooring, N2, Tz, tlow, thigh):
    mooring = bandpass_ctd(mooring, tlow=tlow, thigh=thigh)
    mooring = bandpass_cm(mooring, tlow=tlow, thigh=thigh)
    mooring = bandpass_adcp(mooring, tlow=tlow, thigh=thigh)
    mooring = calc_eta(mooring, Tz)
    vmodes, hmodes, modes = calc_modes(mooring, N2)
    beta_eta = project_eta_on_modes(mooring, modes)
    binbpu, binbpv = downsample_adcp_data(mooring.adcp)
    beta_u, beta_v = project_vels_on_modes(mooring, binbpu, binbpv, hmodes, N2)
    pp = calculate_pressure_perturbation(mooring, beta_eta, vmodes, N2)
    up, vp = calculate_up_vp(mooring, beta_u, beta_v, hmodes)
    Fu, Fv = calculate_ni_flux(up, vp, pp, N2)
    return Fu, Fv


def climatology_woce(mooring):

    # load WOCE data
    woce = gv.ocean.woce_climatology()
    wprf = woce.sel(lon=mooring.lon + 360, lat=mooring.lat, method="nearest").squeeze()

    # buoyancy frequency
    SA = gsw.SA_from_SP(wprf.s, wprf.p, wprf.lon, wprf.lat)
    CT = gsw.CT_from_t(SA, wprf.t, wprf.p)
    N2, pmid = gsw.Nsquared(SA, CT, wprf.p, lat=wprf.lat)
    ni = np.isfinite(N2)
    N2 = N2[ni]
    N2z = -1 * gsw.z_from_p(pmid[ni], mooring.lat)
    # extend constant to bottom
    if np.max(np.abs(N2z)) < mooring.depth:
        N2z = np.append(N2z, np.abs(mooring.depth))
        N2 = np.append(N2, N2[-1])
    if N2z[0] > 0:
        N2z = np.insert(N2z, 0, 0)
        N2 = np.insert(N2, 0, N2[0])
    N2 = xr.DataArray(N2, coords={"depth": (["depth"], N2z)}, dims=["depth"])
    # Interpolate to depth vector with constant dz
    zmax = mooring.depth + 10 - mooring.depth % 10
    znew = np.arange(0, zmax, 10)
    N2 = N2.interp(depth=znew)
    N2.attrs = dict(long_name="N$^2$", units="1/s$^2$")
    N2.depth.attrs = dict(long_name="depth", units="m")
    # temperature gradient
    Tz = wprf.th.differentiate("depth")
    Tz = Tz.where(np.isfinite(Tz), drop=True)
    if Tz.depth.max() < mooring.depth:
        Tzd = Tz.isel(depth=-1)
        Tzd.values = Tz[-1].data
        Tzd["depth"] = mooring.depth
        Tz = xr.concat([Tz, Tzd], dim="depth")
    if Tz.depth.min() > 0:
        Tzs = Tz.isel(depth=0)
        Tzs.values = Tz[0].data
        Tzs["depth"] = 0
        Tz = xr.concat([Tzs, Tz], dim="depth")
    # zi = np.flatnonzero((Tz.z <= 3000) & (np.isnan(Tz.data)))
    # deeptz = Tz.where(((Tz.z <= 3000) & (Tz.z > 2000))).mean()
    # Tz[zi] = deeptz
    Tz.name = "Tz"
    Tz = Tz.interp(depth=znew)
    Tz.attrs = dict(long_name="T$_{z}$", units="°/m")
    Tz.depth.attrs = dict(long_name="depth", units="m")
    return N2, Tz


def climatology_argo_woce(mooring):
    """N2 and Tz for mooring location from WOCE Argo climatology.

    Parameters
    ----------
    mooring : osnap.Mooring
        Mooring structure

    Returns
    -------
    N2 : xarray.DataArray
        Profile of buoyancy frequency squared, from surface to mooring bottom depth.
    Tz : xarray.DataArray
        Vertical temperature gradient profile from surface to mooring bottom depth.

    Notes
    -----
    - Data are extended to surface and seafloor (with depth from mooring structure).

    - Profiles are sorted into a stable state prior to calculating N2 and Tz.

    - Temperature gradient is calculated with z increasing towards the seafloor,
    i.e. it has the wrong sign.

    """
    argo = gv.ocean.woce_argo_profile(mooring.lon, mooring.lat)
    argo.coords["month"] = (
        ["time"],
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    # calculate pressure from depth
    argo["p"] = (["z"], gsw.p_from_z(-argo.z, mooring.lat))
    argo = argo.transpose("z", "time")
    # bring pressure to same dimensions
    _, argo["p"] = xr.broadcast(argo.s, argo.p)
    # calculate absolute salinity
    argo["SA"] = (["z", "time"], gsw.SA_from_SP(argo.s, argo.p, argo.lon, argo.lat))
    # calculate conservative temperature
    argo["CT"] = (["z", "time"], gsw.CT_from_t(argo.SA, argo.t, argo.p))
    # potential density
    argo["sg0"] = (["z", "time"], gsw.sigma0(argo.SA, argo.CT))

    # now calculate N^2 after sorting by density. a climatology shouldn't be
    # unstable anyways!
    N2s = np.zeros((argo.t.shape[0] - 1, argo.t.shape[1])) * np.nan
    N2s = xr.DataArray(data=N2s, dims=["z", "time"])
    N2s.coords["time"] = argo.time
    for i, (g, argoi) in enumerate(argo.groupby("time")):
        argois = argoi.sortby("sg0")
        argois["z"] = argoi.z
        ptmp = gsw.p_from_z(-argoi.z, lat=mooring.lat)
        N2, pmid = gsw.Nsquared(argois.SA, argois.CT, ptmp, lat=mooring.lat)
        N2z = -gsw.z_from_p(pmid, lat=mooring.lat)
        N2s[:, i] = N2
    N2s.coords["z"] = N2z
    N2s = N2s.where(np.isfinite(N2s), drop=True)
    N2s = N2s.where(N2s > 0, np.nan)
    # Extend constant to bottom
    N2deep = xr.full_like(N2s, np.nan)
    N2deep = N2deep.isel(z=-1)
    N2deep["z"] = mooring.depth
    N2deep.values = N2s.isel(z=-1)
    # Extend constant to surface
    N2shallow = N2deep.copy()
    N2shallow["z"] = 0
    N2shallowvalues = N2s.isel(z=0)
    # Bring it all together
    N2s = xr.concat([N2shallow, N2s, N2deep], dim="z")
    N2s = N2s.transpose("z", "time")
    # Get rid of any NaN's
    N2s = N2s.interpolate_na(dim="z")
    # Interpolate to depth vector with constant dz
    zmax = mooring.depth + 10 - mooring.depth % 10
    znew = np.arange(0, zmax, 10)
    N2 = N2s.interp(z=znew)
    N2.attrs = dict(long_name="N$^2$", units="1/s$^2$")
    N2.z.attrs = dict(long_name="depth", units="m")
    N2.name = "N2"

    # temperature gradient
    CT = argo.CT.where(np.isfinite(argo.CT), drop=True)
    tz = []
    for i, (g, CTi) in enumerate(CT.groupby("time")):
        CTs = CTi.sortby(CTi, ascending=False)
        CTs["z"] = CTi.z.data
        tz.append(CTs.differentiate("z"))
    Tz = xr.concat(tz, dim="time")
    Tz.attrs = dict(long_name="T$_{z}$", units="°/m")
    Tz.z.attrs = dict(long_name="depth", units="m")
    Tz = Tz.transpose("z", "time")
    # Extend constant to bottom
    Tzdeep = xr.full_like(Tz, np.nan)
    Tzdeep = Tzdeep.isel(z=-1)
    Tzdeep["z"] = mooring.depth
    Tzdeep.values = Tz.isel(z=-1)
    # Extend constant to surface
    Tzshallow = Tzdeep.copy()
    Tzshallow["z"] = 0
    Tzshallowvalues = Tz.isel(z=0)
    Tz = Tz.interp(z=znew)
    Tz.name = "Tz"

    return N2, Tz


def bandpass_ctd(mooring, tlow, thigh):
    """
    Band-pass filter CTD data (right now theta only)

    Parameters
    ----------
    tlow : float
        cutoff frequency low
    thigh : float
        cutoff frequency high
    """
    thi = mooring.ctd.th.interpolate_na(dim="time", limit=3)
    bpth = _xr_array_clone(thi, "bpth")
    for g, ti in thi.groupby("nomz"):
        ti = ti.squeeze()
        zi = np.squeeze(np.flatnonzero(thi.nomz == g))
        bpth[zi, :] = np.squeeze(bandpass_time_series(ti.data, tlow, thigh))
    mooring.ctd["bpth"] = bpth
    return mooring


def bandpass_cm(mooring, tlow, thigh):
    """
    Band-pass filter current meter u and v.

    Parameters
    ----------
    tlow : float
        cutoff frequency low
    thigh : float
        cutoff frequency high
    """
    ui = mooring.cm.u.interpolate_na(dim="time", limit=3)
    bpu = _xr_array_clone(ui, "bpu")
    for g, u in ui.groupby("nomz"):
        u = u.squeeze()
        zi = np.squeeze(np.flatnonzero(ui.nomz == g))
        bpu[zi, :] = np.squeeze(bandpass_time_series(u.data, tlow, thigh))
    mooring.cm["bpu"] = bpu
    vi = mooring.cm.v.interpolate_na(dim="time", limit=3)
    bpv = _xr_array_clone(vi, "bpv")
    for g, v in vi.groupby("nomz"):
        v = v.squeeze()
        zi = np.squeeze(np.flatnonzero(vi.nomz == g))
        bpv[zi, :] = np.squeeze(bandpass_time_series(v.data, tlow, thigh))
    mooring.cm["bpv"] = bpv
    return mooring


def bandpass_adcp(mooring, tlow, thigh):
    """
    Band-pass filter ADCP u and v.

    Parameters
    ----------
    tlow : float
        cutoff frequency low
    thigh : float
        cutoff frequency high
    """
    ui = mooring.adcp.u.interpolate_na(dim="time", limit=3)
    bpu = _xr_array_clone(ui, "bpu")
    for g, u in ui.groupby("z"):
        u = u.squeeze()
        zi = np.squeeze(np.flatnonzero(ui.z == g))
        bpu[zi, :] = bandpass_time_series(u.data, tlow, thigh)
    mooring.adcp["bpu"] = bpu
    vi = mooring.adcp.v.interpolate_na(dim="time", limit=3)
    bpv = _xr_array_clone(vi, "bpv")
    for g, v in vi.groupby("z"):
        v = v.squeeze()
        zi = np.squeeze(np.flatnonzero(vi.z == g))
        bpv[zi, :] = bandpass_time_series(v.data, tlow, thigh)
    mooring.adcp["bpv"] = bpv
    return mooring


def bandpass_time_series(data, tlow, thigh, minlen=120):
    """
    Band-pass filter time series data.

    Filters chunks of data if there are any NaNs in between.

    Parameters
    ----------
    data : array_like
        Time series data
    tlow : float
        cutoff frequency low
    thigh : float
        cutoff frequency high

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
                bb, lowcut=1 / tlow, highcut=1 / thigh, fs=1, order=2
            )
            lpt[bi[0] : bi[1]] = bpdata
    return lpt


def calc_eta(mooring, Tz):
    """
    Calculate vertical displacement from all CTDs.

    Parameters
    ----------
    Tz : xarray.DataArray
        Vertical temperature gradient with coordinate `z`. Optional second
        dimension `time` if using a seasonal background climatology.
    """
    # Set up data structure
    eta = _xr_array_clone(mooring.ctd.bpth, "eta")
    # loop over all instruments

    for g, ci in mooring.ctd.groupby("nomz"):
        ci = ci.squeeze()
        # Make sure we use the right index in the output matrix
        zi = np.squeeze(np.flatnonzero(mooring.ctd.nomz == g))
        # Exclude NaNs
        ni = np.isfinite(ci.zz)
        # Interpolate temp gradient to instrument (nominal depth)
        # Tzi = interp1d(Tz.z.data, Tz.data)(ci.zz[ni])
        Tzi = Tz.interp(z=ci.nomz, method="linear")
        # Calculate eta from band-passed potential temperature time series
        if "time" in Tzi.dims:
            eta[zi, ni] = ci.bpth[ni] / Tzi[ni].data
        else:
            eta[zi, ni] = ci.bpth[ni] / Tzi
    mooring.eta = eta
    return mooring


def calc_modes(mooring, N2, nmodes=3):
    z = N2.z.data
    N = np.sqrt(N2.data)
    clat = mooring.lat
    calcmodes = nmodes + 1  # vmodes also calculates mode zero...
    Vert, Hori, Edep, PVel = gv.ocean.vmodes(z, N, clat, calcmodes)
    vmodes = Vert[:, 1:calcmodes]
    hmodes = Hori[:, 1:calcmodes]
    modes = xr.Dataset(
        data_vars={
            "vmodes": (["z", "mode"], vmodes),
            "hmodes": (["z", "mode"], hmodes),
        },
        coords={"z": (["z"], N2.z.data), "mode": (["mode"], np.arange(1, calcmodes))},
    )
    return modes


def calc_modes_seasonal(mooring, N2, nmodes=3):
    # first calculate mode for each season
    N2i = []
    assert N2.time.shape[0] == 12
    for g, Ni in N2.groupby("time"):
        vmodes, hmodes, modes = calc_modes(mooring, Ni, nmodes=nmodes)
        N2i.append(modes)
    modes = xr.concat(N2i, dim="time")
    modes.coords["time"] = (["time"], N2.time)
    modes = modes.transpose("z", "mode", "time")
    # now interpolate to mooring timestamps
    modes = interpolate_seasonal_modes_to_mooring(mooring, modes)
    return modes


def project_eta_on_modes(mooring, modes, nmodes=3):
    etaz = mooring.eta.nomz.data
    eta = mooring.eta
    nt = mooring.time.shape[0]
    # note: linear regression returns beta with one extra value, the constant
    # in the equation y = c+mx
    beta_eta = np.full((nt, nmodes), np.nan)
    # seasonal resolving modes
    if "time" in modes.dims:
        for i, ((_, etai), (_, modei)) in enumerate(
            zip(mooring.eta.groupby("time"), modes.groupby("time"))
        ):
            ni = np.flatnonzero(np.isfinite(etai))
            beta_eta[i, :] = project_on_modes(
                etai.data[ni], etaz[ni], modei.vmodes.data, modei.z.data
            )
        # pack beta_hat into DataArray
        beta_eta = beta_to_array(mooring, nmodes, beta_eta)
    elif 0: # new method
        # interpolate modes to depth of eta
        vmeta = modes.vmodes.interp(z=eta.nomz)
        vmeta = vmeta.broadcast_like(eta)
        ni = ~np.isnan(eta)
        ds = xr.Dataset(data_vars={'vmodes': vmeta, 'eta': eta, 'good': ni})
        nt = eta.time.shape[0]
        # pre-allocate array for beta
        ds['beta_eta'] = (['time', 'mode'], np.full((nt, nmodes), np.nan))
        # loop over all time steps
        for i, (g, di) in enumerate(ds.groupby('time')):
            # solve
            y = di.eta.isel(nomz=di.good)
            X = di.vmodes.isel(nomz=di.good)
            X = np.c_[X, np.ones(X.shape[0])]
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            ds['beta_eta'][i, :] = beta_hat[:-1]
    else: # only one set of modes old (wrong?) method
        for i, (g, etai) in enumerate(mooring.eta.groupby("time")):
            ni = np.flatnonzero(np.isfinite(etai))
            beta_eta[i, :] = project_on_modes(
                etai.data[ni], etaz[ni], modes.vmodes.data, modes.z.data
            )
        # pack beta_hat into DataArray
        beta_eta = beta_to_array(mooring, nmodes, beta_eta)
    return beta_eta


def project_vels_on_modes(mooring, binbpu, binbpv, modes, nmodes=3):
    n = len(mooring.time)
    nt = mooring.time.shape[0]
    beta_u = np.full((nt, nmodes), np.nan)
    beta_v = np.full((nt, nmodes), np.nan)
    if "time" in modes.dims:
        for i, (g, modei) in enumerate(modes.groupby("time")):
            u, z = combine_adcp_cm_one_timestep(mooring, binbpu, binbpu.z_bins, i)
            beta_u[i, :] = project_on_modes(u, z, modei.hmodes, modei.z.data)
            v, z = combine_adcp_cm_one_timestep(mooring, binbpv, binbpv.z_bins, i)
            beta_v[i, :] = project_on_modes(v, z, modei.hmodes, modei.z.data)
    else:
        for i in range(nt):
            u, z = combine_adcp_cm_one_timestep(mooring, binbpu, binbpu.z_bins, i)
            beta_u[i, :] = project_on_modes(u, z, modes.hmodes, modes.z.data)

            v, z = combine_adcp_cm_one_timestep(mooring, binbpv, binbpv.z_bins, i)
            beta_v[i, :] = project_on_modes(v, z, modes.hmodes, modes.z.data)
    beta_u = beta_to_array(mooring, nmodes, beta_u)
    beta_v = beta_to_array(mooring, nmodes, beta_v)
    return beta_u, beta_v


def beta_to_array(mooring, nmodes, beta):
    modevec = np.arange(1, nmodes + 1)
    # modevec = np.append(modevec, 0)
    return xr.DataArray(
        beta,
        coords={"time": (["time"], mooring.time), "mode": (["mode"], modevec)},
        dims=["time", "mode"],
    )


def project_on_modes(data, dataz, modes, modesz):
    """
    Project observation onto mode shapes

    Parameters
    ----------
    data : array_like
        Vertical profile of data
    dataz : array_like
        Depth coordinate for data
    modes : array_like (2D)
        Matrix with modes
    modesz : array_like
        Depth coordinate for modes

    Returns
    -------
    beta_hat
        Modal amplitudes for given data profile

    Notes
    -----
    Not returning last item of beta (the c in y = bx + c)
    """
    # interpolate mode to data depths
    vme = interp1d(modesz, modes, bounds_error=False, axis=0)(dataz)
    # solve
    y = data
    X = vme
    X = np.c_[X, np.ones(X.shape[0])]
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta_hat[:-1]


def downsample_adcp_data(adcp):
    print("warning - still need to write good code for downsampling adcp data")
    zbins = np.arange(0, 550, 50)
    zbinlabels = np.arange(25, 525, 50)
    binbpu = adcp.bpu.groupby_bins("z", bins=zbins, labels=zbinlabels).mean()
    binbpv = adcp.bpv.groupby_bins("z", bins=zbins, labels=zbinlabels).mean()
    return binbpu, binbpv


def combine_adcp_cm_one_timestep(mooring, adcpu, adcpz, time):
    # select and downsample ADCP data, then combine with cm
    au = adcpu.isel(time=time).data
    az = adcpz.data
    u = np.concatenate((au, mooring.cm.bpu.isel(time=time).data))
    z = np.concatenate((az, mooring.cm.zz.isel(time=time).data))
    ni = np.isfinite(u)
    return u[ni], z[ni]


def calculate_pressure_perturbation(mooring, beta_eta, modes, N2, nmodes=3):
    # Insert zero into N2 depth vector at the top
    tmp = np.insert(N2.z.data, 0, 0)
    # We are now building the N2 vector such that there should be a zero at
    # the top. Let's make sure this is true:
    # Calculate delta z for N2 depth vector
    N2dz = np.diff(tmp)
    nz = N2.z.shape[0]
    nt = mooring.time.shape[0]
    # Pprime will have shape (nt, nz, nmodes)
    pp = np.full((nt, nz, nmodes), np.nan)
    if "time" in modes.dims:  # Seasonally resolving set of modes, N2
        bvmodes = modes.vmodes.broadcast_like(beta_eta)
        bn2 = N2.broadcast_like(beta_eta)
        for j in range(nmodes):
            pp[:, :, j] = np.cumsum(
                np.multiply(bvmodes.isel(mode=j), beta_eta.isel(mode=j)) * N2 * N2dz,
                axis=1,
            )
        pp -= np.mean(pp, axis=1, keepdims=True)
    else:  # Only one set of modes
        bvmodes = modes.vmodes.broadcast_like(beta_eta)
        for j in range(nmodes):
            pp[:, :, j] = np.cumsum(
                np.multiply(bvmodes.isel(mode=j), beta_eta.isel(mode=j))
                * N2.data
                * N2dz,
                axis=1,
            )
        pp -= np.mean(pp, axis=1, keepdims=True)

    pp = xr.DataArray(data=pp, coords={'time': mooring.time, 'z': modes.z, 'mode': modes.mode}, dims=['time', 'z', 'mode'])

    return pp


def calculate_up_vp(mooring, beta_u, beta_v, modes, nmodes=3):
    nt = mooring.time.shape[0]
    nz = modes.z.shape[0]
    up = np.full((nt, nz, nmodes), np.nan)
    vp = up.copy()
    if "time" in modes.dims:
        bhmodes = modes.hmodes.broadcast_like(beta_u)
        for j in range(nmodes):
            up[:, :, j] = np.multiply(bhmodes.isel(mode=j), beta_u.isel(mode=j))
            vp[:, :, j] = np.multiply(bhmodes.isel(mode=j), beta_v.isel(mode=j))
    else:
        bhmodes = modes.hmodes.broadcast_like(beta_u)
        for j in range(nmodes):
            up[:, :, j] = np.multiply(bhmodes.isel(mode=j), beta_u.isel(mode=j))
            vp[:, :, j] = np.multiply(bhmodes.isel(mode=j), beta_v.isel(mode=j))

    up = xr.DataArray(data=up, coords={'time': mooring.time, 'z': modes.z, 'mode': modes.mode}, dims=['time', 'z', 'mode'])
    vp = xr.DataArray(data=vp, coords={'time': mooring.time, 'z': modes.z, 'mode': modes.mode}, dims=['time', 'z', 'mode'])

    return up, vp


def calculate_ni_flux(up, vp, pp, N2):
    rho0 = 1025
    Fuz = rho0 * up * pp
    Fvz = rho0 * vp * pp
    # Depth-integrated flux
    # Insert zero into N2 depth vector at the top
    tmp = np.insert(N2.z.data, 0, 0)
    # Calculate delta z for N2 depth vector
    N2dz = np.diff(tmp)
    N2dz = np.reshape(N2dz, (1, -1, 1))
    Fu = np.sum(Fuz * N2dz, axis=1)
    Fv = np.sum(Fvz * N2dz, axis=1)
    return Fu, Fv


def plot_mooring_setup(mooring, ax=None):
    if ax is None:
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(2, 4), constrained_layout=True
        )
    ax.plot(
        np.ones_like(mooring.ctd.nomz),
        mooring.ctd.nomz,
        marker="o",
        linestyle="",
        color="#D81B60",
    )
    for ai in mooring.adcplist:
        ax.plot(2, ai.instdep, marker="s", color="#039BE5")
    ax.plot(
        np.ones_like(mooring.cm.nomz) * 3,
        mooring.cm.nomz,
        marker="^",
        linestyle="",
        color="#00897B",
    )
    ax.plot([0, 4], [mooring.depth, mooring.depth], "k-")
    ax.set(xlim=[0, 4])
    ax.set_xticklabels(labels="")


def interpolate_seasonal_data_to_mooring(mooring, da):
    """
    Interpolate seasonal data (once per month) to mooring time series.time

    Parameters
    ----------
    mooring : Mooring
        Mooring dataset.
    da : xr.DataArray
        Seasonal time series to interpolate. Must have dimensions
        `time` and `z`.

    Returns
    -------
    xr.DataArray
        Seasonal data linearly interpolated to mooring time series.
    """
    time = mooring.ctd.time
    tn = time.shape
    start_year = time[0].dt.year.data
    end_year = time[-1].dt.year.data
    years = end_year - start_year + 1
    startstr = "{}-01-15".format(start_year)
    endstr = "{}-12-15".format(end_year)
    timenew = pd.date_range(start=startstr, end=endstr, periods=12 * years)
    tmp = da.data
    timeax = 0 if da.shape[0] == 12 else 1
    assert da.dims[timeax] == "time"
    tmp2 = np.concatenate([tmp, tmp, tmp], timeax)
    da2 = xr.DataArray(
        tmp2,
        coords={"time": (["time"], timenew), "z": (["z"], da.z.data)},
        dims=["z", "time"],
    )
    return da2.interp_like(time)


def interpolate_seasonal_modes_to_mooring(mooring, modes):
    # vmodes
    vmodes = []
    for g, modei in modes.vmodes.groupby("mode"):
        vmodes.append(interpolate_seasonal_data_to_mooring(mooring, modei))
    vmodes = xr.concat(vmodes, dim="mode")
    vmodes.coords["mode"] = (["mode"], modes.mode)
    # hmodes
    hmodes = []
    for g, modei in modes.hmodes.groupby("mode"):
        hmodes.append(interpolate_seasonal_data_to_mooring(mooring, modei))
    hmodes = xr.concat(hmodes, dim="mode")
    hmodes.coords["mode"] = (["mode"], modes.mode)
    # combine
    modes_out = xr.Dataset({"vmodes": vmodes, "hmodes": hmodes})
    modes_out = modes_out.transpose("z", "mode", "time")
    return modes_out


def reconstruct_eta(flux):
    print('warning - may not work for ARGO')
    nz = flux.N2.z.shape[0]
    nt = flux.mooring.time.shape[0]
    nmodes = flux.nmodes
    etam = []
    bvmodes = flux.modes.vmodes.broadcast_like(flux.beta_eta)
    for j in range(nmodes):
        etam.append(np.multiply(bvmodes.isel(mode=j),
                                    flux.beta_eta.isel(mode=j)))
    etam = xr.concat(etam, dim='mode')
    etam.coords['mode'] = flux.modes.mode
    return etam


def plot_eta_modes_time_series(flux):
    etam = reconstruct_eta(flux)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6),
                       constrained_layout=True, sharex=True)
    etam.sum(dim='mode').plot(y='z', yincrease=False, ax=ax[1],
                            cbar_kwargs={'label': r'$\eta$ from modes'})
    gv.plot.concise_date(ax[0])
    flux.mooring.eta.plot(y='nomz', yincrease=False, ax=ax[0],
                          cbar_kwargs={'label': r'$\eta$ observed'})
    gv.plot.concise_date(ax[1])
    ax[1].set(title='')
    ax[0].set(xlabel='')


def plot_eta_modes_one_time_step(flux, ti, etam=None):
    if etam is None:
        etam = reconstruct_eta(flux)
    etams = etam.isel(time=ti).sum(dim='mode')
    fig, ax = gv.plot.quickfig(w=4)
    flux.mooring.eta.isel(time=ti).plot(y='nomz', linestyle='', marker='o', color='0.2')
    etam.isel(time=ti).plot(y='z', hue='mode', color='0.8', add_legend=False)
    etam.isel(time=ti, mode=range(2)).plot(y='z', hue='mode',
                                            color='pink', add_legend=False)

    etams.plot(y='z', color='0.2')
    # (etams-etams.mean(dim='z')).plot(y='z', color='0.2', linestyle='--')
    ax.invert_yaxis()


def plot_up_modes_time_series(flux):
    fig, ax = gv.plot.quickfig(h=3.5, w=6)
    flux.up.sum(dim='mode').plot(y='z', yincrease=False,
                            cbar_kwargs={'label': r'$u^\prime$ from modes',
                                         'shrink': 0.7},)
    gv.plot.concise_date(ax)
    ax.set(xlabel='')


def plot_up_one_time_step(flux, ti):
    binbpu, binbpv = downsample_adcp_data(flux.mooring.adcp)
    u, z = combine_adcp_cm_one_timestep(flux.mooring, binbpu,
                                        binbpu.z_bins, ti)
    fig, ax = gv.plot.quickfig(yi=False, w=4)
    ax.plot(u, z, 'ko')
    flux.up.isel(time=ti).plot(y='z', hue='mode', color='0.8')
    flux.up.isel(time=ti).sum(dim='mode').plot(y='z', color='0.2')
    gv.plot.xsym()


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


def _xr_array_clone(da, name="new"):
    dac = da.copy()
    dac.values = np.full_like(da, np.nan)
    dac = dac.rename(name)
    return dac
