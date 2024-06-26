from setuptools import setup, find_packages
from pathlib import Path

scripts = ['LL_per_photon.py', 'init_emc.py', 'logR.py', 'calculate_probabilities.py', 'update_I.py']

setup(
    name                 = "EMC",
    version              = "2024.0",
    packages             = find_packages(),
    scripts              = ['emc/' + s for s in scripts]
    )
