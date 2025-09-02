"""
Read and write data.
"""

import pathlib
from pathlib import Path
import collections.abc
import numpy as np
import yaml
from box import Box
import gvpy as gv
import getpass
import motuclient, motu_utils
import logging
import pandas as pd
import xarray as xr
import gsw


def load_config() -> Box:
    """Load the yaml config file.

    Returns
    -------
    config : Box
        Config parameters dictionary with dot access.
    """

    def find_config_file():
        parents = list(Path.cwd().parents)
        for pi in parents:
            if pi.as_posix().endswith("niskine"):
                files = list(pi.glob("config.yml"))
                if len(files) == 1:
                    cfile = files[0]
                    root_dir = pi
        return cfile, root_dir

    configfile, root_dir = find_config_file()
    with open(configfile, "r") as ymlfile:
        config = Box(yaml.safe_load(ymlfile))

    # Convert paths to Path objects
    config.path.root = root_dir
    config.path.data = config.path.root.joinpath(config.path.data)
    config.path.fig = config.path.root.joinpath(config.path.fig)

    def replace_variable(dict_in, var, replacement_path):
        d = dict_in.copy()
        n = len(var)
        for k, v in d.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = replace_variable(d.get(k, {}), var, replacement_path)
            elif isinstance(v, str):
                if v.startswith(var + "/"):
                    d[k] = replacement_path.joinpath(v[n + 1 :])
            else:
                d[k] = v
        return d

    # Replace variables from the yaml file.
    config = replace_variable(config, "$data", config.path.data)

    return config


def print_config(print_values=False):
    config = load_config()
    gv.misc.pretty_print(config, print_values=print_values)


def png(fname, subdir=None, **kwargs):
    """Save figure as png to the path defined in config.yml.

    Parameters
    ----------
    fname : str
        Figure name without file extension.
    """
    cfg = load_config()
    if subdir is not None:
        figdir = cfg.path.fig.joinpath(subdir)
    else:
        figdir = cfg.path.fig
    gv.plot.png(fname, figdir=figdir, **kwargs)


def pdf(fname, subdir=None, **kwargs):
    """Save figure as pdf to the path defined in config.yml.

    Parameters
    ----------
    fname : str
        Figure name without file extension.
    """
    cfg = load_config()
    if subdir is not None:
        figdir = cfg.path.fig.joinpath(subdir)
    else:
        figdir = cfg.path.fig
    gv.plot.pdf(fname, figdir=figdir, **kwargs)


def link_proc_adcp(mooringdir):
    """Link processed ADCP data files into package data directory.

    Parameters
    ----------
    mooringdir : str or pathlib.Path
        Directory with mooring data (NISKINE19 on kipapa) that contains
        directories M1, M2 and M3. Links to the directory defined in the config
        file as data.proc.adcp.
    """
    conf = load_config()

    if type(mooringdir) == str:
        mooringdir = Path(mooringdir)

    # Create directory where files will be linked to.
    conf.data.proc.adcp.mkdir(exist_ok=True)

    ADCPS = _adcp_mooring_config()

    for mooring, adcps in ADCPS.items():
        for adcp in adcps:
            package_adcp_dir = conf.data.proc.adcp
            file = (
                mooringdir.joinpath(mooring)
                .joinpath("ADCP")
                .joinpath("proc")
                .joinpath(f"SN{adcp}")
                .joinpath(f"{mooring}_{adcp}.nc")
            )
            link_file = package_adcp_dir.joinpath(file.name)
            if file.exists():
                try:
                    link_file.symlink_to(file)
                except:
                    pass

    # Link the Flowquest .mat file
    file = mooringdir.joinpath(
        "M2/ADCP/proc/FQ10185/fq_converted/FQ_InterpolatedFinal.mat"
    )
    link_file = package_adcp_dir.joinpath(file.name)
    if not link_file.exists():
        link_file.symlink_to(file)

    # # While we are at it, let's also link the mooring location file.
    # loc_file = Path(conf.mooring_locations)
    # if not loc_file.exists():
    #     loc_file.symlink_to(mooringdir.joinpath(loc_file.name))


def link_proc_temperature(mooringdir):
    """Link processed thermistor data files into package data directory.

    Parameters
    ----------
    mooringdir : str or pathlib.Path
        Directory with mooring data (NISKINE19 on kipapa) that contains
        directories M1, M2 and M3. Links to the directory defined in the config
        file as data.proc.thermistor.
    """
    conf = load_config()

    if type(mooringdir) == str:
        mooringdir = Path(mooringdir)

    # Create directory where files will be linked to.
    conf.data.proc.thermistor.mkdir(exist_ok=True)

    # RBR Solos
    rbr_dir = mooringdir.joinpath("M1").joinpath("RBRSolo").joinpath("proc")
    files = rbr_dir.glob("*.nc")
    for file in files:
        link_file = conf.data.proc.thermistor.joinpath(f"{file.name[:6]}.nc")
        if not link_file.exists():
            link_file.symlink_to(file)

    # SBE56
    # SBE05600425_2020-10-08.nc
    sbe_dir = mooringdir.joinpath("M1").joinpath("SBE56").joinpath("proc")
    files = sbe_dir.glob("*.nc")
    for file in files:
        link_file = conf.data.proc.thermistor.joinpath(f"{int(file.name[6:11]):06d}.nc")
        if not link_file.exists():
            link_file.symlink_to(file)


def load_ssh(hourly=False):
    """Load altimeter data.

    Parameters
    ----------
    hourly : bool, optional
        Set to True to load the hourly model data. Defaults to False (daily
        data).

    Returns
    -------
    ssh : xr.Dataset
        SSH data.
    """

    conf = load_config()
    if hourly:
        ssh_files = sorted(conf.data.ssh.glob("hourly_ssh*"))
        all_hourly = [xr.open_dataset(file) for file in ssh_files]
        [ssh.close() for ssh in all_hourly]
        ssh = xr.concat(all_hourly, dim="time")
        ssh = ssh.squeeze().drop("depth")
    else:
        ssh_file = conf.data.ssh.joinpath("mercator_ssh.nc")
        ssh = xr.open_dataset(ssh_file)
        ssh.close()
    ssh = ssh.rename({"longitude": "lon", "latitude": "lat"})
    return ssh


def load_wind_era5():
    """Load ERA5 10m wind data.

    Returns
    -------
    xr.Dataset
        ERA5 10m wind.
    """
    conf = load_config()
    return xr.open_dataset(conf.data.wind.era5)


def load_thermistor(sn):
    cfg = load_config()
    file = cfg.data.proc.thermistor.joinpath(f"{sn:06}.nc")
    return xr.open_dataarray(file)


def load_adcp(mooring=1, sn=None):
    conf = load_config()
    ADCPS = _adcp_mooring_config()
    if sn is None:
        adcps = []
        for sni in ADCPS[f"M{mooring}"]:
            adcp.append(
                xr.open_dataset(
                    conf.data.proc.adcp.joinpath(f"M{mooring}_{sni}.nc"),
                    engine="netcdf4",
                )
            )
        return adcps
    else:
        return xr.open_dataset(conf.data.proc.adcp.joinpath(f"M{mooring}_{sn}.nc"))


def load_gridded_adcp(mooring=1):
    conf = load_config()
    return xr.open_dataset(
        conf.data.gridded.adcp.joinpath(
            f"M{mooring}_gridded_simple_merge_gaps_filled.nc"
        )
    )


def load_mld():
    cfg = load_config()
    mld = xr.open_dataarray(cfg.data.ml.mld)
    return mld


class RetrieveMercatorData:
    def __init__(self, dataset):
        """Retrieve Copernicus data (SSH, Mercator Model).

        Parameters
        ----------
        dataset : {'hourly', 'ssh'}
            Pick the data source. Note that the hourly Mercator data is only
            available starting 2020-01-01.
        """
        self.niskine_config = load_config()
        self.dataset = dataset
        self.USERNAME, self.PASSWORD = self._mercator_credentials()
        self.parameters = self._generate_parameters_dict()

    def _generate_parameters_dict(self):
        """Generate default parameters.

        Returns
        -------
        parameters : dict
            Default parameters
        """
        dataset_parameters = dict(
            hourly={
                "service_id": "GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS",
                "product_id": "global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh",
                "variable": ["thetao", "uo", "vo", "zos"],
                "depth_min": 0.493,
                "depth_max": 0.4942,
                "motu": "https://nrt.cmems-du.eu/motu-web/Motu",
            },
            ssh={
                "service_id": "SEALEVEL_GLO_PHY_L4_MY_008_047-TDS",
                "product_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
                "variable": [
                    "sla",
                    "ugos",
                    "vgos",
                    "ugosa",
                    "vgosa",
                    "err_ugosa",
                    "err_vgosa",
                ],
                "motu": "https://my.cmems-du.eu/motu-web/Motu",
            },
        )

        parameters = {
            "date_min": "2020-05-20 12:00:00",
            "date_max": "2020-05-22 12:00:00",
            "longitude_min": -30.0,
            "longitude_max": -15.0,
            "latitude_min": 65.0,
            "latitude_max": 55.0,
            "out_dir": self.niskine_config.data.ssh.as_posix(),
            "out_name": "test.nc",
            "auth_mode": "cas",
            "user": self.USERNAME,
            "pwd": self.PASSWORD,
        }
        # make output directory if it doesn't exist yet
        self.niskine_config.data.ssh.mkdir(exist_ok=True)

        for k, v in dataset_parameters[self.dataset].items():
            parameters[k] = v

        return parameters

    def _parse_parameters_dict(self, options_dict):
        """Update parameters dict.

        Parameters
        ----------
        options_dict : dict
            Parameters to be updated.
        """
        for k, v in options_dict.items():
            self.parameters[k] = v

    def retrieve_data(self, options_dict):
        """Retrieve Copernicus data.

        Parameters
        ----------
        options_dict : dict
            Default parameters will be updated from the options provided here.
        """
        self._parse_parameters_dict(options_dict)
        motuclient.motu_api.execute_request(_MotuOptions(self.parameters))

    def change_dataset(self, dataset):
        """Switch to 'ssh' or 'hourly' dataset.

        Parameters
        ----------
        dataset : {'hourly', 'ssh'}
        """
        self.dataset = dataset
        self.parameters = self._generate_parameters_dict()

    def _mercator_credentials(self):
        """Read Mercator credentials.

        Will ask for username and password the first time around and store them
        in a hidden file in the base directory that will be ignored by
        .gitignore.
        """
        mercator_credentials_file = self.niskine_config.path.root.joinpath(
            ".mercator_credentials"
        )
        if mercator_credentials_file.exists():
            with open(mercator_credentials_file) as file:
                username, password = [line.rstrip() for line in file]
        else:
            print("sign up for a user account at https://marine.copernicus.eu/")
            print("and provide your credentials here.")
            username = input("Enter your username: ")
            password = getpass.getpass("Enter your password: ")
            with open(mercator_credentials_file, "w") as file:
                for word in [username, password]:
                    file.write(f"{word}\n")
        return username, password


class _MotuOptions:
    """Convert a dictionary into an object with all values as attributes.

    Needed by RetrieveMercatorData().
    """

    def __init__(self, attrs: dict):
        super(_MotuOptions, self).__setattr__("attrs", attrs)

    def __setattr__(self, k, v):
        self.attrs[k] = v

    def __getattr__(self, k):
        try:
            return self.attrs[k]
        except KeyError:
            return None


def _adcp_mooring_config():
    ADCPS = dict(
        M1=[
            3109,
            9408,
            13481,
            14408,
            22476,
        ],
        M2=[
            3110,
            8063,
            8065,
            10219,
            22479,
            23615,
        ],
        M3=[
            344,
            8122,
            12733,
            15339,
            15694,
        ],
    )
    return ADCPS


def mooring_start_end_time(mooring=1):
    timeslice = dict(
        m1=slice(
            np.datetime64("2019-05-17 16:00:00"),
            np.datetime64("2020-10-05 09:00:00"),
        ),
        m2=slice(
            np.datetime64("2019-05-16 15:00:00"),
            np.datetime64("2020-10-05 15:00:00"),
        ),
        m3=slice(
            np.datetime64("2019-05-15 18:00:00"),
            np.datetime64("2020-10-06 09:00:00"),
        ),
    )
    return timeslice[f"m{mooring}"]


def mooring_location(mooring=1):
    conf = load_config()
    locs = xr.open_dataset(conf.mooring_locations)
    loci = locs.sel(mooring=mooring)
    return loci.lon_actual.item(), loci.lat_actual.item(), loci.depth_actual.item()


def read_m1_sensor_config():
    """Read sensor config for mooring M1.

    Returns
    -------
    TODO

    """
    cfg = load_config()
    mm = pd.read_csv(cfg.m1_sensor_config, index_col="SN")
    return mm


def load_microcat(sn):
    cfg = load_config()
    dir = cfg.data.proc.sbe37
    return xr.open_dataset(dir.joinpath(f"SN{sn}/SBE37_{sn}_NISKINE.nc"))


def read_microcats(common_time=None):
    """Read all processed microcat data and merge into one `xr.Dataset`.

    Parameters
    ----------
    common_time : array-like, optional
        Common time vector of type `np.datetime64` that data will be
        interpolated to for merging. Defaults to the full deployment period of
        mooring M1 at 20min interval.

    Returns
    -------
    mc : xr.Dataset
        Merged Microcat data.
    """
    if common_time is None:
        timeslice = mooring_start_end_time(mooring=1)
        common_time = np.arange(
            timeslice.start, timeslice.stop, dtype="datetime64[20m]"
        ).astype("datetime64[ns]")
    cfg = load_config()
    dir = cfg.data.proc.sbe37
    sbes = []
    sbesn = []
    for d in dir.glob("SN*"):
        sn = d.name.split("N")[1]
        sbesn.append(sn)
        dd = list(d.glob("*.nc"))
        sbes.append(xr.open_dataset(dd[0]).interp(time=common_time))
    mc = xr.concat(sbes, "sn")
    for sbei in sbes:
        sbei.close()
    sbesn = [int(sni) for sni in sbesn]
    mc.coords["sn"] = (("sn"), sbesn)
    # sort by mean pressure
    mc["mean_pressure"] = mc.p.mean(dim="time")
    mc = mc.sortby("mean_pressure")

    return mc


def read_chipod_summary(file):
    """Read chipod summary file and return as `xr.Dataset`.

    Parameters
    ----------
    file : pathlib.PosixPath
        Path to summary file.

    Returns
    -------
    chipod : xr.Dataset
        Data from summary file.
    """
    tmp = gv.io.loadmat(file)
    chipod = gv.io.mat2dataset(tmp)
    # convert PSI to dbar
    chipod.attrs["sn"] = tmp["chipod"]
    atm = 14.29
    chipod["p"] = ("time", (chipod["P"].data - atm) / 1.47)
    chipod = chipod.drop("P")
    # get rid of data with spurious time steps
    chipod = chipod.sel(time=chipod.time < np.datetime64("2022"))
    return chipod


def old_mld_to_nc():
    """Convert Anna's mixed layer depth time series at M1 from mat to netcdf."""
    conf = load_config()
    # Load mixed layer depth as calculated by Anna.
    try:
        mldmat = gv.io.loadmat(conf.data.ml.mld_mat)
    except:
        print("Anna's mixed layer depth file needs to live under data/ml/")
    # Generate xarray DataArray
    mld = xr.DataArray(
        data=mldmat.mld,
        coords=dict(time=gv.time.mattime_to_datetime64(mldmat.time)),
        dims=["time"],
        name="mld",
    )
    # Interpolate over nans
    mld = mld.interpolate_na(dim="time")
    # Save to netcdf format
    print(f"Saving mixed layer depth time series to\n{conf.data.ml.mld}")
    mld.to_netcdf(conf.data.ml.mld_old)
    return mld

CFG = load_config()
