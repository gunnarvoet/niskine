"""
Read and write data.
"""

from pathlib import Path
import collections.abc
import yaml
from box import Box
import gvpy as gv
import getpass
import motuclient, motu_utils
import logging
import xarray as xr


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
    conf.data.proc.adcp.mkdir(exist_ok=True)
    for mooring, adcps in ADCPS.items():
        for adcp in adcps:
            package_adcp_dir = conf.data.proc.adcp
            file = (mooringdir.joinpath(mooring)
                    .joinpath('ADCP')
                    .joinpath('proc')
                    .joinpath(f'SN{adcp}')
                    .joinpath(f'{mooring}_{adcp}.nc'))
            link_file = package_adcp_dir.joinpath(file.name)
            if file.exists():
                try:
                    link_file.symlink_to(file)
                except:
                    pass


def load_adcp(mooring=1, sn=None):
    conf = load_config()
    if sn is None:
        adcps = []
        for sni in ADCPS[f'M{mooring}']:
            adcp.append(xr.open_dataset(conf.data.proc.adcp.joinpath(f'M{mooring}_{sni}.nc'), engine='netcdf4'))
        return adcps
    else:
        return xr.open_dataset(conf.data.proc.adcp.joinpath(f'M{mooring}_{sn}.nc'))


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


ADCPS = dict(
    M1 = [3109, 9408, 13481, 14408, 22476, ],
    M2 = [3110, 8063, 8065, 10219, 22479, 23615, ],
    M3 = [344, 8122, 12733, 15339, 15694, ],
        )
