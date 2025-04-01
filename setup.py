import configparser
import os
import setuptools
import sys

# use petsctools to bootstrap the install
sys.path.append(".")
import petsctools
petsctools.init(bootstrap=True)


def write_config():
    config = configparser.ConfigParser()
    config["settings"] = {}

    petsc_dir, petsc_arch = petsctools.get_config()
    config["settings"]["petsc_dir"] = petsc_dir
    config["settings"]["petsc_arch"] = petsc_arch

    with open("petsctools/config.ini", "w") as f:
      config.write(f)


write_config()
setuptools.setup()
