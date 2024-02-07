# Part of 'ljfit' package
"""
Process output from PSI4, GDMA, and ORCA.
"""

#############################################

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Tuple
from .. import __version__


def check_sapt_complete(file: str | Path) -> bool:
    """Check if all SAPT calculations are complete.

    Parameters
    ----------
    file : str | Path
        path to the file

    Returns
    -------
    bool
        True if calculation is complete, False otherwise
    """
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if "*** Psi4 exiting successfully. Buy a developer a beer!" in lines[-1]:
            return True
        else:
            print(f"{file} is not complete!")
            return False


def get_sapt_energy(file: Union[str, Path], sapt_type: str = "ssapt0") -> List[float]:
    """Extract SAPT energies from a PSI4 output file.

    Parameters
    ----------
    file : str | Path
        SAPT output file
    sapt_type : str, optional
        Which type of SAPT, can be sSAPT0 (default) or standard SAPT0, by default "ssapt0"

    Returns
    -------
    [float, float, float, float, float]
        SAPT energies in mEh: electrostatics, exchange, induction, dispersion, total
    """

    e_elst = e_exch = e_ind = e_disp = e_tot = 0.0

    # open file and read relevant lines
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

        # sSAPT0
        if sapt_type == "ssapt0":
            # find the lines with the ssapt0 energies
            for j, line in enumerate(lines):
                if "Electrostatics sSAPT0" in line:
                    e_elst = float(line.split()[2])
                if "Exchange sSAPT0" in line:
                    e_exch = float(line.split()[2])
                if "Induction sSAPT0" in line:
                    e_ind = float(line.split()[2])
                if "Dispersion sSAPT0" in line:
                    e_disp = float(line.split()[2])
                if "Total sSAPT0" in line:
                    e_tot = float(line.split()[2])

        # SAPT0
        elif sapt_type == "sapt0":
            # find the lines with the sapt0 energies
            for j, line in enumerate(lines):
                if "Electrostatics      " in line:
                    e_elst = float(line.split()[1])
                if "Exchange      " in line:
                    e_exch = float(line.split()[1])
                if "Induction      " in line:
                    e_ind = float(line.split()[1])
                if "Dispersion      " in line:
                    e_disp = float(line.split()[1])
                if "Total SAPT0" in line:
                    e_tot = float(line.split()[2])

        # unknown SAPT type
        else:
            raise ValueError("Unknown SAPT type: " + sapt_type)

        # check if all energies were found
        if 0.0 in [e_elst, e_exch, e_ind, e_disp, e_tot]:
            raise ValueError(f"Not all energies were found in the file {file}")

        return [e_elst, e_exch, e_ind, e_disp, e_tot]


def get_coords_from_sapt(file: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read monomer geometries from an SAPT output file.

    Parameters
    ----------
    file : str | Path
        File path to the SAPT output file.

    Returns
    -------
    [np.ndarray, np.ndarray]
        2D arrays of monomer geometries.
    """

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # create empty lists
        l_mon_a = []
        l_mon_b = []

        # initialize variables
        start_line_a = start_line_b = 1e6
        end_line_a = end_line_b = -1

        # get start and end line of monomers
        for i, line in enumerate(lines):
            if "monomer A" in line:
                start_line_a = i + 2
            if "--" in line:
                end_line_a = i - 1
            if "monomer B" in line:
                start_line_b = i + 2
            if "units angstrom" in line:
                end_line_b = i - 2
                break

        # get monomer geometries
        for i, line in enumerate(lines):
            if i >= start_line_a and i <= end_line_a:
                l_mon_a.append(line.split()[1:4])
            if i >= start_line_b and i <= end_line_b:
                l_mon_b.append(line.split()[1:4])

        # transform the lists to np arrays
        ar_mon_a = np.array(l_mon_a, dtype=float)
        ar_mon_b = np.array(l_mon_b, dtype=float)

    return ar_mon_a, ar_mon_b


def get_xyz_from_sapt(file: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read monomer geometries from an SAPT output file.

    Parameters
    ----------
    file : str | Path
        File path to the SAPT output file.

    Returns
    -------
    [pd.DataFrame, pd.DataFrame]
        Dataframes of monomer geometries, including atom names.
    """

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # create empty lists
        l_mon_a = []
        l_mon_b = []

        # initialize variables
        start_line_a = start_line_b = 1e6
        end_line_a = end_line_b = -1

        # get start and end line of monomers
        for i, line in enumerate(lines):
            if "monomer A" in line:
                start_line_a = i + 2
            if "--" in line:
                end_line_a = i - 1
            if "monomer B" in line:
                start_line_b = i + 2
            if "units angstrom" in line:
                end_line_b = i - 2
                break

        # get monomer geometries
        for i, line in enumerate(lines):
            if i >= start_line_a and i <= end_line_a:
                l_mon_a.append(
                    [line.split()[0], line.split()[1], line.split()[2], line.split()[3]]
                )
            if i >= start_line_b and i <= end_line_b:
                l_mon_b.append(
                    [line.split()[0], line.split()[1], line.split()[2], line.split()[3]]
                )

        # transform the lists to np arrays
        df_mon_a = pd.DataFrame(l_mon_a, columns=["atom", "x", "y", "z"])
        df_mon_b = pd.DataFrame(l_mon_b, columns=["atom", "x", "y", "z"])
        # replace element symbols by atom names for ff
        # BF4
        # if atom order is B, F, F, F, F, replace with B, FB, FB, FB, FB
        if df_mon_a["atom"][0] == "B":
            df_mon_a["atom"] = ["B", "FB", "FB", "FB", "FB"]
        # C1c1im
        if df_mon_a["atom"][0] == "C":
            # order is: CW, CW, NA, CR, C1, NA, HCW, HCW, HCR, C1, 6x HC
            df_mon_a["atom"] = [
                "CW",
                "CW",
                "NA",
                "CR",
                "C1",
                "NA",
                "HCW",
                "HCW",
                "HCR",
                "C1",
                "HC",
                "HC",
                "HC",
                "HC",
                "HC",
                "HC",
            ]
        # CC/DC/Graphene
        if df_mon_b["atom"][0] == "C":
            df_mon_b["atom"] = df_mon_b["atom"].replace("C", "CG")
            df_mon_b["atom"] = df_mon_b["atom"].replace("H", "HG")
        # define the x, y, z columns as float
        df_mon_a[["x", "y", "z"]] = df_mon_a[["x", "y", "z"]].astype(float)
        df_mon_b[["x", "y", "z"]] = df_mon_b[["x", "y", "z"]].astype(float)

    return df_mon_a, df_mon_b


def get_charges_from_gdma(file: str | Path) -> np.ndarray:
    """Read the atomic charges from a GDMA output file.

    Parameters
    ----------
    file : str | Path
        File path to the GDMA output file.

    Returns
    -------
    np.ndarray
        Array of atomic charges.
    """

    # open file and read lines
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # create empty list
        l_charges = []

        for i, line in enumerate(lines):
            # find start of new atom section
            if "x =" in line:

                # initialize variables
                q00 = 0.0

                # spherical tensor quantities
                # charge
                if "Q00" in lines[i + 2]:
                    q00 = float(lines[i + 2].split()[2])

                # transform spherical tensor quantities to cartesian tensor quantities
                # charge
                chrg = q00

                # append the values to the np array
                l_charges.append(chrg)

                # go to next line
                continue

            # find end of multipole section
            if "Total multipoles referred" in line:
                break

        # transform the list to a np array
        ar_charges = np.array(l_charges, dtype=float)
        return ar_charges


def get_dipoles_from_gdma(file: str | Path) -> np.ndarray:
    """Read the atomic dipole moments from a GDMA output file.

    Parameters
    ----------
    file : str | Path
        File path to the GDMA output file.

    Returns
    -------
    np.ndarray
        2D array of atomic dipole moments.
    """

    # open file and read lines
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # create empty list
        l_dipoles = []

        for i, line in enumerate(lines):
            # find start of new atom section
            if "x =" in line:

                # initialize variables
                q10 = q11c = q11s = 0.0

                # spherical tensor quantities
                # dipole moment
                if "|Q1| =" in lines[i + 3]:
                    for k, seq in enumerate(lines[i + 3].split()):
                        if seq == "Q1":
                            q10 = float(lines[i + 3].split()[k + 2])
                        if seq == "Q11c":
                            q11c = float(lines[i + 3].split()[k + 2])
                        if seq == "Q11s":
                            q11s = float(lines[i + 3].split()[k + 2])

                # transform spherical tensor quantities to cartesian tensor quantities
                # dipole moment
                mu_z = q10
                mu_x = q11c
                mu_y = q11s

                # append the values to the np array
                l_dipoles.append([mu_x, mu_y, mu_z])

                # go to next line
                continue

            # find end of multipole section
            if "Total multipoles referred" in line:
                break

        # transform the list to a np array
        ar_dipoles = np.array(l_dipoles, dtype=float)
        return ar_dipoles


def get_quadrupoles_from_gdma(file: str | Path) -> np.ndarray:
    """Read the atomic dipole moments from a GDMA output file.

    Parameters
    ----------
    file : str | Path
        File path to the GDMA output file.

    Returns
    -------
    np.ndarray
        3D array of atomic quadrupole moments.
    """

    # open file and read lines
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # create empty list
        l_quadrupoles = []

        for i, line in enumerate(lines):
            # find start of new atom section
            if "x =" in line:

                # initialize variables
                q20 = q21c = q21s = q22c = q22s = 0.0

                # spherical tensor quantities
                # quadrupole moment
                if "|Q2| =" in lines[i + 4]:
                    if not "|Q3| =" in lines[i + 5]:
                        search_line = lines[i + 4].split() + lines[i + 5].split()
                    else:
                        search_line = lines[i + 4].split()
                    for k, seq in enumerate(search_line):
                        if seq == "Q20":
                            q20 = float(search_line[k + 2])
                        if seq == "Q21c":
                            q21c = float(search_line[k + 2])
                        if seq == "Q21s":
                            q21s = float(search_line[k + 2])
                        if seq == "Q22c":
                            q22c = float(search_line[k + 2])
                        if seq == "Q22s":
                            q22s = float(search_line[k + 2])

                # transform spherical tensor quantities to cartesian tensor quantities
                # quadrupole moment
                Q_xx = 0.5 * (-q20 + np.sqrt(3) * q22c)
                Q_yy = 0.5 * (-q20 - np.sqrt(3) * q22c)
                Q_zz = q20
                Q_xy = 0.5 * np.sqrt(3) * q22s
                Q_xz = 0.5 * np.sqrt(3) * q21c
                Q_yz = 0.5 * np.sqrt(3) * q21s
                Q_yx = Q_xy
                Q_zx = Q_xz
                Q_zy = Q_yz

                # append the values to the np array
                l_quadrupoles.append(
                    [[Q_xx, Q_xy, Q_xz], [Q_yx, Q_yy, Q_yz], [Q_zx, Q_zy, Q_zz]]
                )

                # go to next line
                continue

            # find end of multipole section
            if "Total multipoles referred" in line:
                break

        # transform the list to a np array
        ar_quadrupoles = np.array(l_quadrupoles, dtype=float)
        return ar_quadrupoles


def get_chelpg_from_orca(file: str | Path) -> np.ndarray:
    """Extract ChelpG charges from an ORCA output file.

    Parameters
    ----------
    file : str | Path
        File path to the ORCA output file.

    Returns
    -------
    np.ndarray
        Array of atomic charges.
    """

    # open file and read lines
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # create empty list
        l_charges = []

        start_line = 1e6
        end_line = -1

        for i, line in enumerate(lines):
            # find start chelpg section
            if "CHELPG Charges     " in line:
                start_line = i + 2

            # find end of chelpg section
            if "Total charge:" in line:
                end_line = i - 2
                break
        for i, line in enumerate(lines):
            if i >= start_line and i <= end_line:
                l_charges.append(float(line.split()[3]))

        # transform the list to a np array
        ar_charges = np.array(l_charges, dtype=float)
        return ar_charges
