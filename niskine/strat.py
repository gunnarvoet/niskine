"""
Stratification calculations. Broadly this includes:
    - some knockdown analysis
    - depth gridding of thermistor data
"""

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import gsw

import gvpy as gv

import niskine


class MooringConfig:

    """M1 mooring configuration"""

    m = niskine.io.read_m1_sensor_config()
    # drop the chipod that has bad pressure
    m = m.drop(index=614)

    def __init__(self):
        """Read config from spreadsheet"""
        # self.config = niskine.io.read_m1_sensor_config()
        self.config = self.m
        self.sn_pressure = (
            self.m.where(
                (self.m.sensor == "chi")
                | (self.m.sensor == "ctd")
                | (self.m.sensor == "adcp")
            )
            .dropna(how="all")
            .index.to_numpy()
        )
        self.pressure = self.m.where(
            (self.m.sensor == "chi")
            | (self.m.sensor == "ctd")
            | (self.m.sensor == "adcp")
        ).dropna(how="all")
        self.microcat = self.m.where(self.m.sensor == "ctd").dropna(how="all")
        self.microcat_adcp = self.m.where(
            (self.m.sensor == "ctd") | (self.m.sensor == "adcp")
        ).dropna(how="all")
        self.sn_thermistor = (
            self.m.where(self.m.sensor == "t").dropna(how="all").index.to_numpy()
        )
        self.sn_microcat = (
            self.m.where(self.m.sensor == "ctd").dropna(how="all").index.to_numpy()
        )
        self.nomz_thermistor = (
            self.m.where(self.m.sensor == "t").dropna(how="all").depth.to_numpy()
        )

    def info(self, sn):
        return self.m.loc[sn]


def read_chipod_pressure(common_time):
    """Read pressure time series from chipod summary files and interpolate to
    common time vector.

    Parameters
    ----------
    common_time : array-like
        Common time vector with `np.datetime64` type for interpolation.

    Returns
    -------
    p : xr.DataArray

    """
    cfg = niskine.io.load_config()

    files = sorted(cfg.data.input.chipod.glob("*.mat"))

    allp = []
    sn = []
    for file in files:
        c = niskine.io.read_chipod_summary(file)
        p = c.p.interp(time=common_time)
        p = p.expand_dims("sn")
        p.coords["sn"] = (("sn"), [int(c.sn)])
        allp.append(p)
        sn.append(c.sn)
        cc = xr.concat(allp, dim="sn")

    # Let's get rid of one chipod with bad pressure.
    cc = cc.sel(sn=cc.mean(dim="time") > 0)

    return cc


def plot_microcat_pressure_and_thermistor_depths():
    """Plot M1 SBE37 pressure time series and depths of thermistors."""
    s = niskine.io.read_microcats()

    fig, ax = gv.plot.quickfig(h=3, w=7, grid=False)
    # plot thermistor depths
    ax.plot(
        np.tile(np.datetime64("2019-07-01"), len(M.nomz_thermistor)),
        gsw.p_from_z(-M.nomz_thermistor, LAT),
        marker=".",
        markersize=1,
        linestyle="",
    )
    # plot microcat pressure
    s.p.isel(sn=range(5)).gv.plot(
        hue="sn", linewidth=0.5, yincrease=False, ax=ax, color="0.2"
    )
    for g, dsi in s.isel(sn=range(5)).groupby("mean_pressure"):
        dsi = dsi.p.where(~np.isnan(dsi.p), drop=True)
        ax.text(
            dsi.time.isel(time=range(500)).mean().data - np.timedelta64(10, "D"),
            dsi.isel(time=range(500)).mean().data,
            f"{dsi.sn.item()}",
            color=f"0.2",
            va="center",
            ha="right",
            fontsize=9,
            fontweight="normal",
            #         backgroundcolor="w",
        )
    xlim = ax.get_xlim()
    ax.set(xlim=[np.datetime64("2019-03-23"), np.datetime64("2020-11")])


def plot_pressure_and_thermistor_depths():
    """Plot M1 SBE37 (and ADCP) pressure time series and depths of thermistors."""
    p = gather_pressure_time_series()

    fig, ax = gv.plot.quickfig(h=3, w=7, grid=False)
    # plot thermistor depths
    ax.plot(
        np.tile(np.datetime64("2019-07-01"), len(M.nomz_thermistor)),
        gsw.p_from_z(-M.nomz_thermistor, LAT),
        marker=".",
        markersize=1,
        linestyle="",
    )
    # plot pressure time series
    p.isel(sn=range(6)).gv.plot(
        hue="sn", linewidth=0.25, yincrease=False, ax=ax, color="0.2"
    )
    # add labels
    for g, dsi in p.isel(sn=range(6)).groupby("mean_pressure"):
        dsi = dsi.where(~np.isnan(dsi), drop=True)
        ax.text(
            dsi.time.isel(time=range(500)).mean().data - np.timedelta64(10, "D"),
            dsi.isel(time=range(500)).mean().data,
            f"{dsi.sn.item()}",
            color=f"0.2",
            va="center",
            ha="right",
            fontsize=7,
            fontweight="normal",
        )
    xlim = ax.get_xlim()
    ax.set(xlim=[np.datetime64("2019-03-23"), np.datetime64("2020-11")])


def plot_microcat_pressure_segment(time):
    """Plot M1 SBE37 pressure time series for give time period. Omit the
    deepest one.

    Parameters
    ----------
    time : str or slice
        Timespan (anything that can go into selecting a time span using
        `xarray`'s `sel()`.
    """
    fig, ax = gv.plot.quickfig(h=3, w=6, grid=False)
    s = niskine.io.read_microcats()
    ds = s.sel(time=time)
    ds.isel(sn=range(4)).p.gv.plot(
        hue="sn", linewidth=0.7, yincrease=False, ax=ax, color="0.2"
    )
    for g, dsi in ds.isel(sn=range(4)).groupby("mean_pressure"):
        ax.text(
            dsi.p.time.isel(time=-1).data + np.timedelta64(4, "h"),
            dsi.p.isel(time=-1).data,
            f"{dsi.sn.item()}",
            color=f"0.2",
            va="center",
            fontsize=9,
            fontweight="normal",
        )


def knockdown_analysis(sn1, sn2):
    fig, ax = gv.plot.quickfig()
    knockdown, sensor_distance = plot_knockdown_vs_diff(sn1, sn2, ax)
    pp = fit_knockdown(knockdown, sensor_distance)
    plot_knockdown_fit(pp, ax)
    return pp


def plot_knockdown_vs_diff(sn1, sn2, ax):
    s = gather_pressure_time_series()
    if sn2 == "bottom":
        snhigh, snlow = "bottom", sn1
        pl = s.sel(sn=sn1)
        ph = gsw.p_from_z(-2881, LAT)
        diffp = ph - pl
    else:
        mp1 = s.sel(sn=sn1).mean_pressure.item()
        mp2 = s.sel(sn=sn2).mean_pressure.item()
        snhigh, snlow = [sn1, sn2] if mp1 > mp2 else [sn2, sn1]
        ph = s.sel(sn=snhigh)
        pl = s.sel(sn=snlow)
    diffp = ph - pl
    baseline = np.percentile(pl.where(~np.isnan(pl), drop=True), 0.2)
    max_diff = np.percentile(diffp.where(~np.isnan(diffp), drop=True), 99.8)
    y, x = diffp - max_diff, pl - baseline
    ax.plot(x, y, linestyle="", marker=".", color="k", alpha=0.1)
    if sn2 == "bottom":
        ax.set(
            ylabel="normalized distance between sensors",
            xlabel="knockdown",
            title=f"SN{snlow}@{M.info(snlow).depth:1.0f}m - bottom@2881m",
        )
    else:
        ax.set(
            ylabel="normalized distance between sensors",
            xlabel="knockdown",
            title=f"SN{snlow}@{M.info(snlow).depth:1.0f}m - SN{snhigh}@{M.info(snhigh).depth:1.0f}m",
        )
    return x, y


def fit_knockdown(knockdown, sensor_distance):
    mask = (knockdown > 0) & (~np.isnan(sensor_distance))
    pp = np.polynomial.Chebyshev.fit(knockdown[mask], sensor_distance[mask], 2)
    return pp


def knockdown_analysis_absolute(sn1, sn2):
    """Fit pressure to measured distance to neighboring sensor.

    Parameters
    ----------
    sn1 : int
        Serial number of sensor that will be fitted.
    sn2 : int
        Serial number of other sensor.

    Returns
    -------
    pp : np.polynomial.chebyshev.Chebyshep
        Fit object.
    """
    s = gather_pressure_time_series()
    p1 = s.sel(sn=sn1)
    p2 = s.sel(sn=sn2)
    diffp = p1 - p2
    pp = fit_knockdown_absolute(p1, diffp)
    return pp


def fit_knockdown_absolute(pressure, sensor_distance):
    mask = (pressure > 0) & (~np.isnan(sensor_distance))
    pp = np.polynomial.Chebyshev.fit(pressure[mask], sensor_distance[mask], 2)
    return pp


def extend_pressure_based_on_fit(sn1, sn2):
    pp = knockdown_analysis_absolute(sn2, sn1)
    s = gather_pressure_time_series()
    p1 = s.sel(sn=sn1)
    p2 = s.sel(sn=sn2)
    fig, ax = gv.plot.quickfig()
    p2.plot(color="0.4", linewidth=0.5)
    ext = p2 - pp(p2)
    ext.plot(color="C3", linewidth=0.5)
    p1.plot(color="C0", linewidth=0.5)
    ax.invert_yaxis()
    gv.plot.concise_date(ax)
    ax.set(title=f"extend {sn1} based on {sn2}")

    return ext


def plot_knockdown_fit(pp, ax):
    xx, yy = pp.linspace()
    ax.plot(xx, yy, color="r", linewidth=2)


def get_sensor_info(sn):
    return M.info(sn)


def infer_thermistor_depth(sn, sp):

    sp = sp.drop_vars("mean_pressure")
    # sensor info for all pressure time series (excluding chipods) shallower
    # than 1000m nominal depth
    PMA = M.m.where(
        ((M.m.sensor == "ctd") | (M.m.sensor == "adcp")) & (M.m.depth < 1000)
    ).dropna(how="all")

    # info for this specific thermistor
    mt = get_sensor_info(sn)

    # find (nominal) depth of shallowest and deepest pressure sensor
    p_shallowest_depth = PMA.depth.min()
    p_deepest_depth = PMA.depth.max()

    # find SNs of neighboring pressure time series
    # unless sensor is too deep or too shallow
    if mt.depth < p_shallowest_depth:
        # Case: Thermistor shallower than shallowest pressure sensor. Subtract
        # distance to pressure sensor from its depth measurement.
        case = "shallow"
        psn = PMA.depth.idxmin()
        m = get_sensor_info(psn)
        # nominal depths and distance along mooring line
        tdist = m.depth - mt.depth
        # calculate depths
        p1 = sp.sel(sn=psn)
        d1 = -gsw.z_from_p(p1, LAT)
        sensor_depth = d1 - tdist

    elif mt.depth > p_deepest_depth:
        # Case: Temperature sensor deeper than the deepest pressure sensor
        # (except for the one right near the bottom, but that one doesn't tell
        # us much about knockdown). Here we can't scale the distance below.
        # Instead, we'll calculate knockdown, scale that with the ratio of
        # distance from pressure sensor to pressure sensor distance from
        # bottom, and add it to the nominal depth.
        case = "deep"
        psn = PMA.depth.idxmax()
        m = get_sensor_info(psn)
        # nominal distance to pressure sensor
        tdist = mt.depth - m.depth
        # nominal pressure sensor distance to bottom
        pdist = 2881 - m.depth
        # Scale factor for knockdown (knockdown reduces linearly as we approach
        # the bottom).
        ratio = 1 - tdist / pdist
        # calculate depths
        p1 = sp.sel(sn=psn)
        d1 = -gsw.z_from_p(p1, LAT)
        knockdown = d1 - np.percentile(d1, 0.5)
        sensor_depth = mt.depth + knockdown * ratio

    else:
        case = "regular"
        psn_upper = PMA.where(PMA.depth < mt.depth).depth.idxmax()
        m_upper = get_sensor_info(psn_upper)
        psn_lower = PMA.where(PMA.depth > mt.depth).depth.idxmin()
        m_lower = get_sensor_info(psn_lower)

        # nominal depths
        pdist = m_lower.depth - m_upper.depth
        tdist1 = mt.depth - m_upper.depth
        tdist2 = m_lower.depth - mt.depth

        ratio = tdist1 / pdist

        # calculate depths
        p1 = sp.sel(sn=psn_upper)
        p2 = sp.sel(sn=psn_lower)
        d1 = -gsw.z_from_p(p1, LAT)
        d2 = -gsw.z_from_p(p2, LAT)
        ddiff = d2 - d1
        sensor_depth = d1 + ddiff * ratio

    sensor_depth.name = "depth"
    sensor_depth["sn"] = mt.name

    return sensor_depth


def gather_pressure_time_series(common_time=None):
    if common_time is None:
        timeslice = niskine.io.mooring_start_end_time(mooring=1)
        common_time = np.arange(
            timeslice.start, timeslice.stop, dtype="datetime64[20m]"
        ).astype("datetime64[m]")
    # microcats
    s = niskine.io.read_microcats(common_time=common_time)
    sp = s.p
    sp["mean_pressure"] = sp.mean(dim="time")
    sp = sp.sortby("mean_pressure")
    # adcp
    adcp = niskine.io.load_adcp(mooring=1, sn=13481)
    adcp_pressure_offset = 10
    adcp_pressure = adcp.pressure.interp(time=common_time) + adcp_pressure_offset
    adcp_pressure.coords["sn"] = 13481
    adcp_pressure["mean_pressure"] = adcp_pressure.mean(dim="time")
    # merge
    p = xr.concat([sp, adcp_pressure], dim="sn")
    p = p.sortby("mean_pressure")

    return p


# We'll be dealing with mooring M1 here so let's have location and depth handy
LON, LAT, DEPTH = niskine.io.mooring_location(mooring=1)

# Load mooring configuration
M = MooringConfig()
