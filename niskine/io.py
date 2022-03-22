"""
Read and write data.
"""

from pathlib import Path
import collections.abc
import yaml
from box import Box
import gvpy as gv


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


def load_m1_adcp():
    pass
