# Part of 'ljfit' package
"""
Define important parameters for the system, such as the monomer names and the orientations to be considered.
Use these parameters to calculate the interaction energy between the monomers.
"""

#############################################

from __future__ import annotations

from pathlib import Path


from .. import __version__

from .calc_energy import extract_energy


def get_system_energy(
    system: str,
) -> None:
    """Get the interaction energy between two monomers, and derive further energies.

    Parameters
    ----------
    system : str
        The name of the system to be considered.
    """

    # define the systems
    calc_cc_bf4 = system in ["all", "cc-bf4", "bf4", "cc"]
    calc_cc_c1c1im = system in ["all", "cc-c1c1im", "c1c1im", "cc"]
    calc_dc_bf4 = system in ["all", "dc-bf4", "bf4", "dc"]
    calc_dc_c1c1im = system in ["all", "dc-c1c1im", "c1c1im", "dc"]

    if calc_cc_bf4:
        extract_energy(
            monomer_a="bf4", monomer_b="cc", orientation=["c-face", "c-tip", "c-edge1"]
        )
    if calc_cc_c1c1im:
        extract_energy(
            monomer_a="c1c1im",
            monomer_b="cc",
            orientation=["c-cor", "c-h21", "c-h22"],
        )
    if calc_dc_bf4:
        extract_energy(
            monomer_a="bf4", monomer_b="dc", orientation=["c-face", "c-tip", "c-edge1"]
        )
    if calc_dc_c1c1im:
        extract_energy(
            monomer_a="c1c1im",
            monomer_b="dc",
            orientation=["c-cor", "c-h21", "c-h22"],
        )
