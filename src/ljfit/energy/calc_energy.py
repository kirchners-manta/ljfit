# Part of 'ljfit' package
"""
Calculate the interaction energy between two monomers, and derive further energies.
"""

#############################################

from __future__ import annotations

from typing import List, Optional, Union

import pandas as pd
from pathlib import Path
import numpy as np


from .. import __version__

from ..helpers import get_file_list

from .process_output import (
    check_sapt_complete,
    get_sapt_energy,
    get_coords_from_sapt,
    get_charges_from_gdma,
    get_dipoles_from_gdma,
    get_quadrupoles_from_gdma,
    get_chelpg_from_orca,
)

from .multipole_expansion import (
    get_e_multipole_truncation,
    get_e_pol,
)


def extract_energy(
    monomer_a: str,
    monomer_b: str,
    orientation: List[str],
) -> None:
    """Extract the SAPT0 energy contributions from PSI4 output files.
    Based on this, calculate the multipole, polarisation and pairwise additive energies.

    Parameters
    ----------
    monomer_a : str
        The short name of the first monomer, used in the file names.
    monomer_b : str
        The short name of the second monomer, used in the file names.
    orientation : List[str]
        The orientation of the monomers to be considered, used in the file names.
    """

    ll_all = [
        get_file_list(
            "./" + monomer_b + "-" + monomer_a + "/",
            orientation[i] + "/[1-9]*/sapt.out",
        )
        for i in range(len(orientation))
    ]

    # check the output files for completeness
    # drop all files that are not complete
    for _, liste in enumerate(ll_all):
        for file in liste.copy():
            check = check_sapt_complete(file)
            if check == False:
                liste.remove(file)

    # multipole moments
    charges_b = get_charges_from_gdma("./gdma/" + monomer_b + "/gdma.out")
    dipoles_b = get_dipoles_from_gdma("./gdma/" + monomer_b + "/gdma.out")
    quadrupoles_b = get_quadrupoles_from_gdma("./gdma/" + monomer_b + "/gdma.out")
    # chelpg charges
    charges_a = get_chelpg_from_orca("./chelpg/" + monomer_a + "/hf-dz/chelpg.out")

    # info
    print(f"Calculating the energies for the {monomer_b}-{monomer_a} systems.")

    # iterate over all lists of files and extract the energies
    for _, liste in enumerate(ll_all):
        l_energy = []
        for _, file in enumerate(liste):
            # read the geometries
            a, b = get_coords_from_sapt(file)
            # dist is the minimum absolute z coordinate of the atoms of monomer A
            dist = np.min(np.abs(a[:, 2]))
            # sSAPT0 energy
            e_elst, e_exch, e_ind, e_disp, e_tot = get_sapt_energy(
                file, sapt_type="ssapt0"
            )
            # multipole energy
            e_chelpg_hf = (
                get_e_multipole_truncation(
                    charges_a, charges_b, dipoles_b, quadrupoles_b, a, b
                )
                * 1000
            )
            # polarization energy
            e_pol = get_e_pol(a, b, charges_a) * 1000
            e_pol_rahul = get_e_pol(a, b, charges_a, alpha=1.8, damping=0.5) * 1000
            # pairwise additive remainder
            e_pair = e_tot - e_chelpg_hf - e_pol

            # append the energies to the list
            l_energy.append(
                [
                    dist,
                    e_elst,
                    e_exch,
                    e_ind,
                    e_disp,
                    e_tot,
                    e_chelpg_hf,
                    e_pol,
                    e_pol_rahul,
                    e_pair,
                ]
            )

        # create a separate dataframe for each liste element and save it to a csv file
        # define path so save the csv file
        outpath_energy = (
            "./"
            + monomer_b
            + "-"
            + monomer_a
            + "/"
            + str(file).split("/")[1]
            + "/e_summary.csv"
        )
        df_energy = pd.DataFrame(
            l_energy,
            columns=[
                "distance",
                "e_elst",
                "e_exch",
                "e_ind",
                "e_disp",
                "e_tot",
                "e_chelpg_hf",
                "e_pol",
                "e_pol_rahul",
                "e_pair",
            ],
        )
        # reorder the df by increasing distance
        df_energy = df_energy.sort_values(by="distance")
        df_energy.to_csv(outpath_energy, sep=";", index=False)

    # info
    print("Finished.")
