"""
Calculate NI fluxes.
"""

import numpy as np
from matplotlib import ticker
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import netCDF4
from scipy.interpolate import interp1d
import warnings
import gsw
import gvpy as gv
import scipy
import pandas as pd

import niskine


class Flux:
    """
    Calculate flux time series for OSNAP mooring.

    Parameters
    ----------
    mooring : OSNAPMooring or NISKINeMooring
        Mooring data structure generated with `mooring.OSNAPMooring()` or
        `mooring.NISKINeMooring()`.
    bandwidth : float
        Bandwidth parameter
    type : str ['NI', 'M2'], optional
        Type of flux, determines center frequency for bandpass filter.
        Default 'NI'. NOT IMPLEMENTED YET.
    climatology : str ['ARGO', 'WOCE'], optional
        Select type of climatology to use for calculation of modes and
        vertical displacements. Default 'ARGO' which resolves the seasonal
        cycle.
    runall : bool, optional
        Set to False to run all processing steps manually. Default True.

    """

    def __init__(
        self, mooring, bandwidth, nmodes=3, type="NI", climatology="WOCE", runall=True, adcp_vertical_resolution=100,
    ):
        self.mooring = mooring
        self.bandwidth = bandwidth
        self.climatology = climatology
        self.nmodes = nmodes
        self.adcp_vertical_resolution = adcp_vertical_resolution
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
        return f"flux calculations for mooring {self.mooring.name}"

    def band(self):
        t = gv.ocean.inertial_period(lat=self.mooring.lat) * 24
        self.thigh = 1 / self.bandwidth * t
        self.tlow = self.bandwidth * t

    def background_gradients(self):
        if self.climatology == "ARGO":
            self.N2s, self.Tzs = niskine.clim.climatology_argo_woce(
                self.mooring.lon, self.mooring.lat, self.mooring.bottom_depth
            )
            # interpolate to mooring time vector
            self.N2 = niskine.clim.interpolate_seasonal_data(
                self.mooring.adcp.time, self.N2s
            )
            self.Tz = niskine.clim.interpolate_seasonal_data(
                self.mooring.adcp.time, self.Tzs
            )
        elif self.climatology == "WOCE":
            self.N2, self.Tz = niskine.clim.climatology_woce(
                self.mooring.lon, self.mooring.lat, self.mooring.bottom_depth
            )
            # For now rename depth back to z. Eventually name it depth everywhere.
            self.N2 = self.N2.rename(depth="z")
            self.Tz = self.Tz.rename(depth="z")

    def find_modes(self):
        if self.climatology == "WOCE":
            self.modes = calc_modes(self.mooring, self.N2, nmodes=self.nmodes)
        elif self.climatology == "ARGO":
            self.modes = calc_modes_seasonal(self.mooring, self.N2s, nmodes=self.nmodes)

    def bandpass(self):
        self.mooring = bandpass_ctd(self.mooring, tlow=self.tlow, thigh=self.thigh)
        if hasattr(self.mooring, "cm"):
            self.mooring = bandpass_cm(self.mooring, tlow=self.tlow, thigh=self.thigh)
        self.mooring = bandpass_adcp(self.mooring, tlow=self.tlow, thigh=self.thigh)

    def eta_modes(self):
        self.mooring = calc_eta(self.mooring, self.Tz)
        self.beta_eta = project_eta_on_modes(
            self.mooring, self.modes, nmodes=self.nmodes
        )

    def vel_modes(self):
        self.binbpu, self.binbpv = downsample_adcp_data(self.mooring.adcp, self.adcp_vertical_resolution)
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
                    self.Fu.data,
                    {"long_name": "NI Flux U", "units": "W/m"},
                ),
                "fy_ni": (
                    ["time", "mode"],
                    self.Fv.data,
                    {"long_name": "NI Flux V", "units": "W/m"},
                ),
            },
            coords={
                "time": (["time"], self.mooring.time.data),
                "mode": (
                    ["mode"],
                    self.modes.mode.data,
                    {"long_name": "mode", "units": ""},
                ),
            },
        )


def near_inertial_energy_flux(mooring, bandwidth, N2, Tz):
    """Calculate near-inertial energy flux time series for one mooring."""
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
    eigenspeed = PVel[1:calcmodes]
    equivalent_depth = Edep[1:calcmodes]
    modes = xr.Dataset(
        data_vars={
            "vmodes": (["z", "mode"], vmodes),
            "hmodes": (["z", "mode"], hmodes),
            "eigenspeed": (["mode"], eigenspeed),
            "equivalent_depth": (["mode"], equivalent_depth),
        },
        coords={"z": (["z"], N2.z.data), "mode": (["mode"], np.arange(1, calcmodes))},
    )
    modes.eigenspeed.attrs = dict(long_name='eigenspeed $c_n$', units='m/s')
    return modes


def calc_modes_seasonal(mooring, N2, nmodes=3):
    # first calculate mode for each season
    N2i = []
    assert N2.time.shape[0] == 12
    for g, Ni in N2.groupby("time"):
        Ni = Ni.squeeze()
        # vmodes, hmodes, modes = calc_modes(mooring, Ni, nmodes=nmodes)
        modes = calc_modes(mooring, Ni, nmodes=nmodes)
        N2i.append(modes)
    modes = xr.concat(N2i, dim="time")
    modes.coords["time"] = (["time"], N2.time.data)
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
            beta_eta[i, :] = project_on_vmodes(
                etai.squeeze().data[ni], etaz[ni], modei.squeeze().vmodes.data, modei.squeeze().z.data
            )
        # pack beta_hat into DataArray
        beta_eta = beta_to_array(mooring, nmodes, beta_eta)
    elif 0:  # new method
        # interpolate modes to depth of eta
        vmeta = modes.vmodes.interp(z=eta.nomz)
        vmeta = vmeta.broadcast_like(eta)
        ni = ~np.isnan(eta)
        ds = xr.Dataset(data_vars={"vmodes": vmeta, "eta": eta, "good": ni})
        nt = eta.time.shape[0]
        # pre-allocate array for beta
        ds["beta_eta"] = (["time", "mode"], np.full((nt, nmodes), np.nan))
        # loop over all time steps
        for i, (g, di) in enumerate(ds.groupby("time")):
            # solve
            y = di.eta.isel(nomz=di.good)
            X = di.vmodes.isel(nomz=di.good)
            X = np.c_[X, np.ones(X.shape[0])]
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            ds["beta_eta"][i, :] = beta_hat[:-1]
    else:  # only one set of modes old (wrong?) method
        for i, (g, etai) in enumerate(mooring.eta.groupby("time")):
            ni = np.flatnonzero(np.isfinite(etai))
            beta_eta[i, :] = project_on_vmodes(
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
            beta_u[i, :] = project_on_modes(u, z, modei.squeeze().hmodes, modei.squeeze().z.data)
            v, z = combine_adcp_cm_one_timestep(mooring, binbpv, binbpv.z_bins, i)
            beta_v[i, :] = project_on_modes(v, z, modei.squeeze().hmodes, modei.squeeze().z.data)
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
        coords={"time": (["time"], mooring.time.data), "mode": (["mode"], modevec)},
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


def project_on_vmodes(data, dataz, modes, modesz):
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
    # X = np.c_[X, np.ones(X.shape[0])]
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta_hat


def downsample_adcp_data(adcp, vertical_resolution=100):
    zbins = np.arange(0, 1500 + vertical_resolution, vertical_resolution)
    zbinlabels = np.arange(vertical_resolution / 2, 1500 + vertical_resolution / 2, vertical_resolution)
    print(f"averaging ADCP data into {vertical_resolution}m vertical bins")
    binbpu = adcp.bpu.groupby_bins("z", bins=zbins, labels=zbinlabels).mean()
    binbpv = adcp.bpv.groupby_bins("z", bins=zbins, labels=zbinlabels).mean()
    return binbpu, binbpv


def combine_adcp_cm_one_timestep(mooring, adcpu, adcpz, time):
    # select and downsample ADCP data, then combine with cm
    au = adcpu.isel(time=time).data
    az = adcpz.data
    if hasattr(mooring, "cm"):
        u = np.concatenate((au, mooring.cm.bpu.isel(time=time).data))
        z = np.concatenate((az, mooring.cm.zz.isel(time=time).data))
    else:
        u = au
        z = az
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

    pp = xr.DataArray(
        data=pp,
        coords={"time": mooring.time, "z": modes.z, "mode": modes.mode},
        dims=["time", "z", "mode"],
    )

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

    up = xr.DataArray(
        data=up,
        coords={"time": mooring.time, "z": modes.z, "mode": modes.mode},
        dims=["time", "z", "mode"],
    )
    vp = xr.DataArray(
        data=vp,
        coords={"time": mooring.time, "z": modes.z, "mode": modes.mode},
        dims=["time", "z", "mode"],
    )

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
        vmodes.append(niskine.clim.interpolate_seasonal_data(mooring.time, modei.squeeze()))
    vmodes = xr.concat(vmodes, dim="mode")
    vmodes.coords["mode"] = (["mode"], modes.mode.data)
    # hmodes
    hmodes = []
    for g, modei in modes.hmodes.groupby("mode"):
        hmodes.append(niskine.clim.interpolate_seasonal_data(mooring.time, modei.squeeze()))
    hmodes = xr.concat(hmodes, dim="mode")
    hmodes.coords["mode"] = (["mode"], modes.mode.data)
    # eigenspeed
    eigenspeed = []
    for g, modei in modes.eigenspeed.groupby("mode"):
        eigenspeed.append(niskine.clim.interpolate_seasonal_data(mooring.time, modei.squeeze()))
    eigenspeed = xr.concat(eigenspeed, dim="mode")
    eigenspeed.coords["mode"] = (["mode"], modes.mode.data)
    # equivalent depth
    equivalent_depth = []
    for g, modei in modes.equivalent_depth.groupby("mode"):
        equivalent_depth.append(niskine.clim.interpolate_seasonal_data(mooring.time, modei.squeeze()))
    equivalent_depth = xr.concat(equivalent_depth, dim="mode")
    equivalent_depth.coords["mode"] = (["mode"], modes.mode.data)
    # combine
    modes_out = xr.Dataset({"vmodes": vmodes, "hmodes": hmodes, "eigenspeed": eigenspeed, "equivalent_depth": equivalent_depth})
    modes_out = modes_out.transpose("z", "mode", "time")
    modes_out.eigenspeed.attrs = dict(long_name='eigenspeed $c_n$', units='m/s')
    return modes_out


def reconstruct_eta(flux):
    nz = flux.N2.z.shape[0]
    nt = flux.mooring.time.shape[0]
    nmodes = flux.nmodes
    etam = []
    bvmodes = flux.modes.vmodes.broadcast_like(flux.beta_eta)
    for j in range(nmodes):
        etam.append(np.multiply(bvmodes.isel(mode=j), flux.beta_eta.isel(mode=j)))
    etam = xr.concat(etam, dim="mode")
    etam.coords["mode"] = flux.modes.mode
    return etam


def plot_eta_modes_time_series(flux):
    etam = reconstruct_eta(flux)
    fig, ax = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 6), constrained_layout=True, sharex=True
    )
    opts = dict(vmin=-30, vmax=30, cmap="RdBu_r")
    etam.sum(dim="mode").plot(
        y="z", yincrease=False, ax=ax[1], cbar_kwargs={"label": r"$\eta$ from modes"}, **opts,
    )
    gv.plot.concise_date(ax[0])
    flux.mooring.eta.plot(
        y="nomz", yincrease=False, ax=ax[0], cbar_kwargs={"label": r"$\eta$ observed"}, **opts,
    )
    gv.plot.concise_date(ax[1])
    ax[1].set(title="")
    ax[0].set(xlabel="")


def plot_eta_modes_one_time_step(flux, ti, etam=None):
    if etam is None:
        etam = reconstruct_eta(flux)
    etams = etam.isel(time=ti).sum(dim="mode")
    fig, ax = gv.plot.quickfig(w=4)
    flux.mooring.eta.isel(time=ti).plot(y="nomz", linestyle="", marker="o", color="0.2")
    etam.isel(time=ti).plot(y="z", hue="mode", color="0.8", add_legend=False)
    etam.isel(time=ti, mode=range(2)).plot(
        y="z", hue="mode", color="pink", add_legend=False
    )

    etams.plot(y="z", color="0.2")
    # (etams-etams.mean(dim='z')).plot(y='z', color='0.2', linestyle='--')
    ax.invert_yaxis()


def plot_up_modes_time_series(flux):
    fig, ax = gv.plot.quickfig(h=3, w=9)
    flux.up.sum(dim="mode").plot(
        y="z",
        yincrease=False,
        cbar_kwargs={"label": r"$u^\prime$ from modes", "shrink": 0.7},
    )
    gv.plot.concise_date(ax)
    ax.set(xlabel="")


def plot_up_one_time_step(flux, ti, nmodes=3):
    u, z = combine_adcp_cm_one_timestep(flux.mooring, flux.binbpu, flux.binbpu.z_bins, ti)

    fig, ax = gv.plot.quickfig(yi=False, w=3.5, grid=True)
    ax.plot(u, z, "ko")

    if nmodes == 3:
        for i, col in enumerate(["C0", "C4", "C6"]):
            flux.up.isel(time=ti, mode=i).plot(y="z", color="w", linewidth=1.75)
            flux.up.isel(time=ti, mode=i).plot(y="z", color=col, linewidth=1, label=f"mode {i+1}")
    else:
        for i in range(nmodes):
            flux.up.isel(time=ti, mode=i).plot(y="z", color="w", linewidth=1.75)
            flux.up.isel(time=ti, mode=i).plot(y="z", linewidth=1, label=f"mode {i+1}")

    flux.up.isel(time=ti).sum(dim="mode").plot(y="z", color="w", linewidth=2)
    flux.up.isel(time=ti).sum(dim="mode").plot(y="z", color="k", linewidth=1.5, label="$\sum$")
    gv.plot.xsym()
    ax.set(ylabel="depth [m]", xlabel="eastward velocity [m/s]", title="")
    ax.legend()
    tstr = gv.time.datetime64_to_str(flux.mooring.time.isel(time=ti), unit="m").replace("T", "\n")
    ax.annotate(tstr, (-0.03, 2900), ha="center", fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))


def plot_both_mode_fits_one_time_step(flux, ti):
    fig, axx = plt.subplots(nrows=1, ncols=2, figsize=(7.5/1.2, 5/1.2),
                       constrained_layout=True, sharey=True)
    for axi in axx:
        gv.plot.axstyle(axi, grid=True)

    linecols = ["C0", "C4", "C6"]

    # Plot velocity / horizontal modes
    u, z = combine_adcp_cm_one_timestep(flux.mooring, flux.binbpu, flux.binbpu.z_bins, ti)
    ax = axx[0]
    ax.plot(u, z, linestyle="", marker="o", markersize=6, color="k", label="obs")
    # plot the deep constraint in gray
    ax.plot(u[-1], z[-1], linestyle="", marker="o", markersize=6, mec="k", mfc="w")
    ax.annotate("zero\nconstraint", xy=(u[-1], z[-1]), xytext=(-0.017, z[-1]-650),
                color="0.3", xycoords="data", ha="center", va="top", backgroundcolor="w",
                arrowprops=dict(color="0.3", arrowstyle="->", shrinkB=7)
                )
    for i, col in enumerate(linecols):
        flux.up.isel(time=ti, mode=i).plot(ax=ax, y="z", color="w", linewidth=2.5)
        flux.up.isel(time=ti, mode=i).plot(ax=ax, y="z", color=col, linewidth=1.5, label=f"mode {i+1}")
    flux.up.isel(time=ti).sum(dim="mode").plot(ax=ax, y="z", color="w", linewidth=2.5)
    flux.up.isel(time=ti).sum(dim="mode").plot(ax=ax, y="z", color="k", linewidth=1.5, label="$\sum$")
    gv.plot.xsym(ax=ax)
    ax.set(ylabel="depth [m]", xlabel="eastward velocity u [m/s]", title="horizontal velocity modes")
    ax.legend(loc=(0.7, 0.3))
    # ax.legend(loc="best")
    tstr = gv.time.datetime64_to_str(flux.mooring.time.isel(time=ti), unit="m").replace("T", "\n")
    # ax.annotate(tstr, (-0.015, 2900), ha="center", fontsize=9)
    ax.annotate(tstr, (0.13, 0.03), ha="center", backgroundcolor="w", xycoords="axes fraction", fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))

    # Plot eta modes
    etam = reconstruct_eta(flux)
    ax = axx[1]
    flux.mooring.eta.isel(time=ti).plot(ax=ax, y="nomz", linestyle="", marker="o", markersize=6, color="k")
    for i, col in enumerate(linecols):
        etam.isel(time=ti, mode=i).plot(ax=ax, y="z", color="w", linewidth=2.5)
        etam.isel(time=ti, mode=i).plot(ax=ax, y="z", color=col, linewidth=1.5)
    etam.isel(time=ti).sum(dim="mode").plot(ax=ax, y="z", color="w", linewidth=2.5)
    etam.isel(time=ti).sum(dim="mode").plot(ax=ax, y="z", color="k", linewidth=1.5)
    gv.plot.xsym(ax=ax)
    ax.set(ylabel="", xlabel="vertical displacement $\eta$ [m]", title="vertical displacement modes")
    ax.invert_yaxis()

    gv.plot.subplotlabel(axx, fs=14, x=0.03, y=0.94)


def plot_flux_time_series(ax, flux, add_legend=True, add_title=True):
    time = flux.mooring.time
    fu, fv = flux.Fu, flux.Fv
    linecols = ["C0", "C4", "C6"]
    for i, col in enumerate(linecols):
        ax.plot(time, np.cumsum(fv.isel(mode=i))*3600/1e9, color="w", linestyle="-", linewidth=1.5)
        ax.plot(time, np.cumsum(fv.isel(mode=i))*3600/1e9, color=col, linestyle="-", linewidth=0.75, label=f"mode {i+1}")
        ax.plot(time, np.cumsum(fu.isel(mode=i))*3600/1e9, color="w", linestyle="-", linewidth=1.5)
        ax.plot(time, np.cumsum(fu.isel(mode=i))*3600/1e9, color=col, linestyle="--", linewidth=0.75)

    ax.plot(time, np.cumsum(np.sum(fv, axis=1))*3600/1e9, color="w", linestyle="-", linewidth=2)
    ax.plot(time, np.cumsum(np.sum(fv, axis=1))*3600/1e9, color="k", linestyle="-", linewidth=1.0, alpha=1.0, label='F$_\mathregular{v}$')

    ax.plot(time, np.cumsum(np.sum(fu, axis=1))*3600/1e9, color="w", linestyle="-", linewidth=2)
    ax.plot(time, np.cumsum(np.sum(fu, axis=1))*3600/1e9, color="k", linestyle="--", alpha=1.0, linewidth=1.0, label='F$_\mathregular{u}$')

    ax.set(ylabel=r'$\int \, \mathregular{F} \mathregular{dt}$ [GJ/m]')
    if add_title:
        ax.set(title=f'{flux.mooring.name} Near-Inertial Low-Mode Energy Flux')
    if add_legend:
        ax.legend()
    gv.plot.concise_date()
    ax.grid()


def flux_mag_and_dir(flux):
    Fu = flux.flux.fx_ni.sum(dim="mode")
    Fv = flux.flux.fy_ni.sum(dim="mode")

    Fuv = Fu + 1j * Fv

    Fmag, Fdir = np.absolute(Fuv), np.angle(Fuv)
    Fdir = Fdir - np.pi / 2
    Fdir[Fdir < 0] = Fdir[Fdir < 0] + 2 * np.pi

    Fmag.attrs["units"] = "W/m"
    Fmag.attrs["long_name"] = "NI low-mode flux magnitude"

    return Fmag, Fdir


def plot_flux_polar(ax, flux):

    Fu = flux.flux.fx_ni.sum(dim="mode")
    Fv = flux.flux.fy_ni.sum(dim="mode")

    mFu = Fu.mean(dim="time")
    mFv = Fv.mean(dim="time")

    Fuv = Fu + 1j * Fv
    # Fuv = mFu + 1j * mFv
    Fmag, Fdir = np.absolute(Fuv), np.angle(Fuv)
    Fdir = Fdir - np.pi / 2
    Fdir[Fdir < 0] = Fdir[Fdir < 0] + 2 * np.pi
    delta_theta = 2 * np.pi / 48
    _, _, _, h = ax.hist2d(
        Fdir,
        Fmag,
        bins=[np.arange(0, 2 * np.pi + delta_theta, delta_theta), np.arange(0, 1250, 50)],
        density=True,
        cmap="RdPu",
        # cmap="BuPu",
        norm=colors.LogNorm(vmin=1e-5, vmax=3e-3),
    )
    plt.colorbar(h, label=r"PDF of F$_\mathregular{NI}$", shrink=0.5, pad=0.14)
    ax.set_yticks([400, 800, 1200])
    ax.set_yticklabels(["400 ", "800 ", "1200 W/m "], color="k", ha='left', va='bottom')
    ax.set_rlabel_position(300)
    ax.set_rmax(1200)
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
    ax.set_xticklabels(["E", "S", "W", "N"])
    ax.grid(which="major", axis="y", color="k")
    ax.set(theta_direction=-1)
    # mFuv = mFu + 1j * mFv
    # mFmag, mFdir = np.absolute(mFuv), np.angle(mFuv)
    # mFdir = mFdir - np.pi / 2
    # mFdir = mFdir + 2 * np.pi if mFdir < 0 else mFdir
    # ax.plot(mFdir, mFmag, 'wx');


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
