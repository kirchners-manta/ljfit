"""
MSDiff
======

This is a simple program designed to help with the generation of force field parameters for molecular simulations.
It is specifically designed to process the outputs of SAPT calculations in PSI4, extract relevant data, and use it to
fit Lennard-Jones parameters to the SAPT data.

This program is developed by Tom Frömbgen, (Group of Prof. Dr. Barbara Kirchner, University of Bonn, Germany).
It is published under the MIT license.
"""

from .__version__ import __version__
from .cli import console_entry_point
from .energy import (
    check_sapt_complete,
    get_sapt_energy,
    get_coords_from_sapt,
    get_xyz_from_sapt,
    get_charges_from_gdma,
    get_dipoles_from_gdma,
    get_quadrupoles_from_gdma,
    get_chelpg_from_orca,
)
from .helpers import get_file_list, read_xyz, write_xyz, combine_xyz
from .lj import get_system_lj_params
