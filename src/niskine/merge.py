"""
Merge ADCP data for a mooring.
"""

from pathlib import Path
import numpy as np
import gvpy as gv
import xarray as xr
import gsw

from . import io


class MergeADCP:

    """Merge ADCPs from a mooring."""

    def __init__(
        self,
        mooring: int,
        dt_min: int = 10,
        dz_m: int = 16,
        min_end_time="2020",
        start=None,
        stop=None,
        method="simple",
        dropna=True,
    ):
        """Merge

        Parameters
        ----------
        mooring : int
            Mooring number.
        dt_min : int, otpional
            Period [minutes] of new time vector. Defaults to 10min.
        dz_m : int, optional
            Depth interval for new depth vector. Defaults to 16m.
        min_end_time : str or None, optional
            Minimum time that an ADCP has to last to be included. Defaults to
            '2020' which excludes all ADCPs that stopped sampling early after
            two weeks. Includes all ADCPs if set to None.
        start : str or np.datetime64 or None, optional
            Start time of time vector. Defaults to None which returns the start
            of the mooring time series.
        stop : str or np.datetime64 or None, optional
            End time of time vector. Defaults to None which returns the end of
            the mooring time series.
        method : {'simple', 'median'}, optional
            Merge method. Defaults to simple.
        dropna : bool, optional
            Drop depth levels with no data, defaults to True.
        """

        self.mooring = mooring
        self.dt_min = dt_min
        self.dz_m = dz_m
        self.min_end_time = min_end_time
        self.start = start
        self.stop = stop
        self.method = method

        self.all_adcps = load_mooring_adcps(self.mooring)
        self.adcps = select_adcps(self.all_adcps, self.min_end_time)
        self.adcps = at_depth_only(self.adcps, self.mooring)
        self.adcps_sorted = sort_in_depth(self.adcps)

        print("sampling periods")
        print_sampling_period(self.adcps_sorted)

        self.tnew = self.generate_time_vector()
        self.znew = self.generate_depth_vector()

        print("interpolating time...")
        self.adcps_sorted_ti = interpolate_time(self.adcps_sorted, self.tnew)

        print("interpolating depth...")
        self.adcps_sorted_ti_zi = interpolate_depth(
            self.adcps_sorted_ti, self.znew
        )

        print("merging...")
        self.pick_merge_method()
        self.merge_adcps()

        if dropna:
            self.merged = _dropna(self.merged)

    def generate_time_vector(self):
        time_span = io.mooring_start_end_time(mooring=self.mooring)
        if self.start is not None:
            start = np.datetime64(self.start)
            time_span = slice(start, time_span.stop)
        if self.stop is not None:
            stop = np.datetime64(self.stop)
            time_span = slice(time_span.start, stop)
        return np.arange(
            time_span.start, time_span.stop, self.dt_min, dtype="datetime64[m]"
        )
        # Run the following line for second precision of the time vector
        # (should not matter...)
        # tnew = np.array([np.datetime64(ti, "s") for ti in tnew])

    def generate_depth_vector(self):
        return np.arange(0, 3000 + self.dz_m, self.dz_m)

    def pick_merge_method(self):
        if self.method == "simple":
            self.merge_fun = simple_merge
        elif self.method == "median":
            self.merge_fun = median_merge

    def merge_adcps(self):
        self.merged = self.merge_fun(self.adcps_sorted_ti_zi)
        self.merged = add_mooring_metadata(self.merged, self.mooring)
        self.merged = add_auxilliary_data(self.adcps_sorted_ti_zi, self.merged)

    def fill_gaps(self):
        self.mergedf = fill_gaps(self.merged)


def load_mooring_adcps(mooring: int) -> list[xr.Dataset]:
    conf = io.load_config()
    files = sorted(conf.data.proc.adcp.glob(f"M{mooring}*.nc"))
    # Read all ADCPs and return as list.
    return [xr.open_dataset(mi) for mi in files]


def select_adcps(adcps: list[xr.Dataset], longer_than) -> list[xr.Dataset]:
    if type(longer_than) == str:
        longer_than = np.datetime64(longer_than)
    if longer_than is not None:
        return [ai for ai in adcps if ai.time.isel(time=-1) > longer_than]
    else:
        return adcps


def at_depth_only(adcps: list[xr.Dataset], mooring: int) -> list[xr.Dataset]:
    time_span = io.mooring_start_end_time(mooring)
    return [ai.sel(time=time_span) for ai in adcps]


def sort_in_depth(adcps: list[xr.Dataset]) -> list[xr.Dataset]:
    mean_p = [ai.pressure.mean().data for ai in adcps]
    return [adcps[i] for i in np.argsort(mean_p)]


def print_sampling_period(adcps: list[xr.Dataset]):
    def find_sampling_period(adcpi):
        dt = adcpi.time.diff(dim="time").median()
        dt = np.timedelta64(dt.data, "s")
        return dt

    [print(ai.attrs["sn"], ":", find_sampling_period(ai)) for ai in adcps]


def interpolate_time(
    adcps: list[xr.Dataset], tnew: np.ndarray
) -> list[xr.Dataset]:
    return [ai.interp(time=tnew) for ai in adcps]


def interpolate_depth(
    adcps: list[xr.Dataset], znew: np.ndarray
) -> list[xr.Dataset]:
    return [ai.interp(z=znew) for ai in adcps]


def determine_overlap(adcps_interp):
    tmp = [ai.u for ai in adcps_interp]
    tmp = xr.concat(tmp, dim="adcp")
    return tmp.where(np.isnan(tmp), other=1)


def simple_merge(adcps_interp):
    # I think we need to copy so we don't change the input data with this
    # operation.
    c = adcps_interp.copy()
    c = [remove_extra_variables(ai) for ai in c]
    for i in range(len(c) - 1):
        c[i + 1] = c[i + 1].combine_first(c[i])
    return c[-1]


def median_merge(adcps_interp):
    adcps_interp = [remove_extra_variables(ai) for ai in adcps_interp]
    tmp = xr.concat(adcps_interp, dim="adcp")
    return tmp.median(dim="adcp", keep_attrs=True)


def spline_merge(adcps_interp):
    pass


def fill_gaps(merged):
    merged = _drop_variable(merged, "w")
    for var in ["u", "v"]:
        merged[var] = merged[var].interpolate_na(dim="z")
    return merged


def remove_extra_variables(merged):
    vars = [
        "pg",
        "e",
        "e_std",
        "u_std",
        "v_std",
        "w_std",
        "amp",
        "temperature",
        "pressure",
        "pressure_std",
        "pressure_max",
        "npings",
    ]
    for var in vars:
        merged = _drop_variable(merged, var)
    return merged


def add_auxilliary_data(adcps_interp, merged):
    zz = [_p_to_depth(ai.pressure, merged.attrs["lat"]) for ai in adcps_interp]
    merged["xducer_depth"] = (("adcp", "time"), zz)
    temp = [ai.temperature for ai in adcps_interp]
    merged["temperature"] = (("adcp", "time"), temp)
    sns = [ai.attrs["sn"] for ai in adcps_interp]
    merged.coords["adcp"] = (("adcp"), sns)
    return merged


def add_mooring_metadata(merged, mooring: int):
    lon, lat, depth = io.mooring_location(1)
    merged.attrs = dict(
        project="NISKINe",
        mooring=f"M{mooring}",
        lon=lon,
        lat=lat,
        bottom_depth=depth,
    )
    return merged


def save_merged(merged, suffix=None):
    conf = io.load_config()
    conf.data.gridded.adcp.mkdir(exist_ok=True, parents=True)
    filename = f"{merged.attrs['mooring']}_gridded.nc"
    savename = conf.data.gridded.adcp.joinpath(filename)
    if suffix is not None:
        savename = savename.parent.joinpath(
            savename.stem + "_" + suffix + savename.suffix
        )
    merged.to_netcdf(
        savename,
    )


def _drop_variable(ds, var):
    return ds.drop(var) if var in ds else ds


def _dropna(ds):
    return ds.dropna(dim="z", how="all")


def _p_to_depth(p, lat):
    z = -gsw.z_from_p(p, lat)
    z.attrs = dict(long_name="depth", units="m")
    return z
