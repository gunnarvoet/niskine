"""
Climatology-related functions.
"""

from pathlib import Path

import gsw
import gvpy as gv
import numpy as np
import xarray as xr
import pandas as pd

import niskine


def climatology_woce(lon, lat, bottom_depth):
    """Extract profiles of N^2 and Tz from WOCE climatology.

    Parameters
    ----------
    lon : float
        Location longitude.
    lat : float
        Location latitude.
    bottom_depth : float
        Bottom depth at location.

    Returns
    -------
    N2 : xarray.DataArray
        Profile of buoyancy frequency squared, from surface to mooring bottom depth.
    Tz : xarray.DataArray
        Vertical temperature gradient profile from surface to mooring bottom depth.

    Notes
    -----
    - Data are extended to surface and seafloor (using bottom depth provided).
    """

    lon = lon + 360 if lon < 0 else lon

    # load WOCE data
    woce = gv.ocean.woce_climatology()
    wprf = woce.sel(lon=lon, lat=lat, method="nearest").squeeze()

    # buoyancy frequency
    SA = gsw.SA_from_SP(wprf.s, wprf.p, wprf.lon, wprf.lat)
    CT = gsw.CT_from_t(SA, wprf.t, wprf.p)
    N2, pmid = gsw.Nsquared(SA, CT, wprf.p, lat=wprf.lat)
    ni = np.isfinite(N2)
    N2 = N2[ni]
    N2z = -1 * gsw.z_from_p(pmid[ni], lat)
    # extend constant to bottom
    if np.max(np.abs(N2z)) < bottom_depth:
        N2z = np.append(N2z, np.abs(bottom_depth))
        N2 = np.append(N2, N2[-1])
    if N2z[0] > 0:
        N2z = np.insert(N2z, 0, 0)
        N2 = np.insert(N2, 0, N2[0])
    N2 = xr.DataArray(N2, coords={"depth": (["depth"], N2z)}, dims=["depth"])
    # Interpolate to depth vector with constant dz
    zmax = bottom_depth + 10 - bottom_depth % 10
    znew = np.arange(0, zmax, 10)
    N2 = N2.interp(depth=znew)
    N2.attrs = dict(long_name="N$^2$", units="1/s$^2$")
    N2.depth.attrs = dict(long_name="depth", units="m")
    # temperature gradient
    Tz = wprf.th.differentiate("depth")
    Tz = Tz.where(np.isfinite(Tz), drop=True)
    if Tz.depth.max() < bottom_depth:
        Tzd = Tz.isel(depth=-1)
        Tzd.values = Tz[-1].data
        Tzd["depth"] = bottom_depth
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


def climatology_argo_woce(lon, lat, bottom_depth):
    """N2 and Tz for mooring location from WOCE Argo climatology.

    Parameters
    ----------
    lon : float
        Mooring longitude.
    lat : float
        Mooring latitude.
    bottom_depth : float
        Bottom depth at mooring site in meters.

    Returns
    -------
    N2 : xarray.DataArray
        Profile of buoyancy frequency squared, from surface to mooring bottom depth.
    Tz : xarray.DataArray
        Vertical temperature gradient profile from surface to mooring bottom depth.

    Notes
    -----
    - Data are extended to surface and seafloor (using bottom depth provided).

    - Profiles are sorted into a stable state prior to calculating N2 and Tz.

    - Temperature gradient is calculated with z increasing towards the seafloor,
    i.e. it has the wrong sign.
    """
    argo = gv.ocean.woce_argo_profile(lon, lat)
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
    # Calculate pressure from depth.
    argo["p"] = (["z"], gsw.p_from_z(-argo.z, lat).data)
    argo = argo.transpose("z", "time")
    # Bring pressure to same dimensions.
    _, argo["p"] = xr.broadcast(argo.s, argo.p)
    # calculate absolute salinity
    argo["SA"] = (["z", "time"], gsw.SA_from_SP(argo.s, argo.p, argo.lon, argo.lat).data)
    # Calculate conservative temperature.
    argo["CT"] = (["z", "time"], gsw.CT_from_t(argo.SA, argo.t, argo.p).data)
    # Potential density.
    argo["sg0"] = (["z", "time"], gsw.sigma0(argo.SA, argo.CT).data)

    # Now calculate N^2 after sorting by density. A climatology shouldn't be
    # unstable anyways!
    N2s = np.zeros((argo.t.shape[0] - 1, argo.t.shape[1])) * np.nan
    N2s = xr.DataArray(data=N2s, dims=["z", "time"])
    N2s.coords["time"] = argo.time
    for i, (g, argoi) in enumerate(argo.groupby("time")):
        argois = argoi.sortby("sg0")
        argois["z"] = argoi.z
        ptmp = gsw.p_from_z(-argoi.z, lat=lat)
        N2, pmid = gsw.Nsquared(argois.SA, argois.CT, ptmp, lat=lat)
        N2z = -gsw.z_from_p(pmid, lat=lat)
        N2s[:, i] = N2
    N2s.coords["z"] = N2z
    N2s = N2s.where(np.isfinite(N2s), drop=True)
    N2s = N2s.where(N2s > 0, np.nan)
    # Extend constant to bottom.
    N2deep = xr.full_like(N2s, np.nan)
    N2deep = N2deep.isel(z=-1)
    N2deep["z"] = bottom_depth
    N2deep.values = N2s.isel(z=-1)
    # Extend constant to surface.
    N2shallow = N2deep.copy()
    N2shallow["z"] = 0
    N2shallowvalues = N2s.isel(z=0)
    # Bring it all together.
    N2s = xr.concat([N2shallow, N2s, N2deep], dim="z")
    N2s = N2s.transpose("z", "time")
    # Get rid of any NaN's.
    N2s = N2s.interpolate_na(dim="z")
    # Interpolate to depth vector with constant dz
    zmax = bottom_depth + 10 - bottom_depth % 10
    znew = np.arange(0, zmax, 10)
    N2 = N2s.interp(z=znew)
    N2.attrs = dict(long_name="N$^2$", units="1/s$^2$")
    N2.z.attrs = dict(long_name="depth", units="m")
    N2.name = "N2"

    # Temperature gradient.
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
    # Extend constant to bottom.
    Tzdeep = xr.full_like(Tz, np.nan)
    Tzdeep = Tzdeep.isel(z=-1)
    Tzdeep["z"] = bottom_depth
    Tzdeep.values = Tz.isel(z=-1)
    # Extend constant to surface.
    Tzshallow = Tzdeep.copy()
    Tzshallow["z"] = 0
    Tzshallowvalues = Tz.isel(z=0)
    Tz = Tz.interp(z=znew)
    Tz.name = "Tz"

    return N2, Tz


def interpolate_seasonal_data(time, da):
    """
    Interpolate seasonal data (once per month) to time series vector.

    Parameters
    ----------
    time : array-like
        Time vector.
    da : xr.DataArray
        Seasonal time series to interpolate. Must have dimensions
        `time` and `z`.

    Returns
    -------
    xr.DataArray
        Seasonal data linearly interpolated to time vector.
    """
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
    # We need to assemble the seasonal data into as many years as needed.
    if years > 1:
        tmp2 = np.concatenate([tmp, tmp], axis=timeax)
    else:
        tmp2 = tmp
    if years > 2:
        yy = years - 2
        while yy > 0:
            tmp2 = np.concatenate([tmp2, tmp], axis=timeax)
            yy -= 1
    da2 = xr.DataArray(
        tmp2,
        coords={"time": (["time"], timenew), "z": (["z"], da.z.data)},
        dims=["z", "time"],
    )
    return da2.interp_like(time)


def get_wkb_factors(adcp):
    m1lon, m1lat, m1depth = niskine.io.mooring_location(mooring=1)
    n2, tz = climatology_argo_woce(m1lon, m1lat, m1depth)
    an2 = niskine.clim.interpolate_seasonal_data(adcp.time, n2)
    adcp["n2"] = an2.interp_like(adcp)
    adcp["N"] = np.sqrt(adcp.n2)
    N0 = adcp.N.where((adcp.z<1200) & (adcp.z>300)).mean().item()
    wkb = 1/np.sqrt(adcp.N/N0)
    wkb.name = "wkb normalization factor"
    return wkb
    wkb.name = "wkb normalization"
