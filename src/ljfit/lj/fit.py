# Part of 'ljfit' package
"""
Fit Lennard-Jones parameters to a system of monomers.
"""

#############################################

from __future__ import annotations

from typing import List, Optional, Union
import itertools
from lmfit import Parameters, minimize, report_fit
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import distance_matrix
from ..helpers import get_file_list, print_lj_params, params_to_df
import matplotlib.pyplot as plt


from .. import __version__
from ..energy import get_xyz_from_sapt

# define constants
EH2KCAL = 627.503  # kcal/mol
ANGSTROM2BOHR = 1.8897259885789  # conversion factor from angstrom to bohr


def get_e_lj(file: str | Path, epsilon: dict, sigma: dict) -> float:
    """Calculate the Lennard-Jones potential between two molecules.

    Parameters
    ----------
    file : str | Path
        File path to the ORCA output file with the sSAPT0 energy.
    epsilon : dict
        Dictionary with the Lennard-Jones epsilon parameters for each atom pair.
        In a.u.
    sigma : dict
        Dictionary with the Lennard-Jones sigma parameters for each atom pair.
        In a.u.

    Returns
    -------
    float
        Lennard-Jones energy in 1e3 a.u.
    """

    # get geometries
    a, b = get_xyz_from_sapt(file)

    # drop all rows in geometry b that are hydrogen atoms
    b = b[~b["atom"].str.contains("H")]

    # convert units
    a[["x", "y", "z"]] *= ANGSTROM2BOHR
    b[["x", "y", "z"]] *= ANGSTROM2BOHR

    # distance matrix
    dist = distance_matrix(a[["x", "y", "z"]], b[["x", "y", "z"]])

    # calculate the Lennard-Jones potential
    e_lj = 0
    for k, atom_a in enumerate(a["atom"]):
        for l, atom_b in enumerate(b["atom"]):
            e_lj += (
                4
                * epsilon[f"{atom_b}_{atom_a}"]
                * (
                    (sigma[f"{atom_b}_{atom_a}"] / dist[k, l]) ** 12
                    - (sigma[f"{atom_b}_{atom_a}"] / dist[k, l]) ** 6
                )
            )

    return e_lj * 1000


def lj_dataset(params: Parameters, i: int, file: str | Path) -> float:
    """Generate parameters for a fit of Lennard-Jones parameters to a dataset.

    Parameters
    ----------
    file : str | Path
        File path to the SAPT input/ output file with the geometries.
    params : Parameters
        Parameters object with the Lennard-Jones parameters to fit.
    i : int
        Index of the dataset.

    Returns
    -------
    float
        Lennard-Jones energy in a.u.
    """

    # extract parameters from the Parameters object
    # and create a dictionary with the parameters
    epsilon = {}
    sigma = {}
    for _, p in enumerate(params):
        if p.endswith(f"_{i}"):
            if p.startswith("epsilon_"):
                epsilon[f'{p.split("_")[1]}_{p.split("_")[2]}'] = params[p].value
            elif p.startswith("sigma_"):
                sigma[f'{p.split("_")[1]}_{p.split("_")[2]}'] = params[p].value

    return get_e_lj(file, epsilon, sigma)


def get_residuals(params: Parameters, filelist: List[Path], data: pd.Series) -> List:
    """Objective function for the Lennard-Jones parameter fitting.

    Parameters
    ----------
    params : Parameters
        Parameters object with the Lennard-Jones parameters to fit.
    filelist : list
        List of file paths to the SAPT input/ output files with the geometries.
    data : pd.Series
        Series with the reference energies.

    Returns
    -------
    np.ndarray
        Flattened 1D array of residuals.
    """
    # number of input files
    ndata = len(filelist)

    residuals = []
    # iterate over all datasets
    for i in range(ndata):
        # read LJ energy from file
        e_pair = data["e_pair"][i]

        residuals.append(lj_dataset(params, i, filelist[i]) - e_pair)

    return residuals


def get_params_to_fit(filelist: List[str | Path]) -> pd.DataFrame:
    """Generate a DataFrame with initial Lennard-Jones parameters for a fit.

    Parameters
    ----------
    filelist : List[str  |  Path]
        List of file paths to the SAPT input/ output files with the geometries.
        Only the first file is used to generate the parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with initial Lennard-Jones parameters for a fit.
        Sigma and epsilon are output in a.u.
    """

    # LJ parameters, taken from the CL&P force field
    epsilon_clap = {  # 'atom_A-atom_B': epsilon in kcal/mol
        "CG-B": 0.34120,  # C(Graphene) - B(BF4-)
        "CG-FB": 0.27339,  # C(Graphene) - F(BF4-)
        "CG-NA": 0.45642,  # C(Graphene) - N(Imidazolium)
        "CG-CR": 0.29288,  # C(Graphene) - CR(Imidazolium)
        "CG-CW": 0.29288,  # C(Graphene) - CW(Imidazolium)
        "CG-C1": 0.28439,  # C(Graphene) - C1(Imidazolium)
        "CG-HCR": 0.19173,  # C(Graphene) - HCR(Imidazolium)
        "CG-HCW": 0.19173,  # C(Graphene) - HCW(Imidazolium)
        "CG-HC": 0.19173,  # C(Graphene) - HC(Imidazolium side chain)
        # no LJ parameters for H atoms on graphene-like structures
        "HG-B": 0.0,  # H(Graphene) - B(BF4-)
        "HG-FB": 0.0,  # H(Graphene) - F(BF4-)
        "HG-NA": 0.0,  # H(Graphene) - N(Imidazolium)
        "HG-CR": 0.0,  # H(Graphene) - CR(Imidazolium)
        "HG-CW": 0.0,  # H(Graphene) - CW(Imidazolium)
        "HG-C1": 0.0,  # H(Graphene) - C1(Imidazolium)
        "HG-HCR": 0.0,  # H(Graphene) - HCR(Imidazolium)
        "HG-HCW": 0.0,  # H(Graphene) - HCW(Imidazolium)
        "HG-HC": 0.0,  # H(Graphene) - HC(Imidazolium side chain)
    }

    sigma_clap = {  # 'atom_A-atom_B': sigma in Angstrom
        "CG-B": 3.565,  # C(Graphene) - B(BF4-)
        "CG-FB": 3.335,  # C(Graphene) - F(BF4-)
        "CG-NA": 3.400,  # C(Graphene) - N(Imidazolium)
        "CG-CR": 3.550,  # C(Graphene) - CR(Imidazolium)
        "CG-CW": 3.550,  # C(Graphene) - CW(Imidazolium)
        "CG-C1": 3.525,  # C(Graphene) - C1(Imidazolium)
        "CG-HCR": 2.985,  # C(Graphene) - HCR(Imidazolium)
        "CG-HCW": 2.985,  # C(Graphene) - HCW(Imidazolium)
        "CG-HC": 3.125,  # C(Graphene) - HC(Imidazolium side chain)
        # no LJ parameters for H atoms on graphene-like structures
        "HG-B": 0.0,  # H(Graphene) - B(BF4-)
        "HG-FB": 0.0,  # H(Graphene) - F(BF4-)
        "HG-NA": 0.0,  # H(Graphene) - N(Imidazolium)
        "HG-CR": 0.0,  # H(Graphene) - CR(Imidazolium)
        "HG-CW": 0.0,  # H(Graphene) - CW(Imidazolium)
        "HG-C1": 0.0,  # H(Graphene) - C1(Imidazolium)
        "HG-HCR": 0.0,  # H(Graphene) - HCR(Imidazolium)
        "HG-HCW": 0.0,  # H(Graphene) - HCW(Imidazolium)
        "HG-HC": 0.0,  # H(Graphene) - HC(Imidazolium side chain)
    }

    # get geometries
    a, b = get_xyz_from_sapt(filelist[0])

    # remove all atoms H from geometry b
    b = b[~b["atom"].str.contains("H")]

    # find all unique atom pairs by atom name
    atom_pairs = list(set(itertools.product(b["atom"], a["atom"])))

    # find the parameters for each atom pair in the CL&P dictionaries
    # form a DataFrame with initial Lennard-Jones parameters for a fit

    l_params = []
    for _, pair in enumerate(atom_pairs):
        # have _ in atom_pair name to make it a valid parameter for lmfit. - in name is not allowed
        l_params.append(
            [
                f"{pair[0]}_{pair[1]}",
                epsilon_clap[f"{pair[0]}-{pair[1]}"] / EH2KCAL,
                sigma_clap[f"{pair[0]}-{pair[1]}"] * ANGSTROM2BOHR,
            ]
        )

    return pd.DataFrame(l_params, columns=["atom_pair", "epsilon", "sigma"])


def plot_fit(
    params: Parameters,
    filelist: List[Path],
    data: pd.DataFrame,
    monomer_a: str,
    monomer_b: str,
    orientation: str,
    i: int,
) -> None:
    """Generate a plot of the current fit.

    Parameters
    ----------
    params : Parameters
        Current parameters of the fit
    filelist : List[Path]
        List of files from which the LJ energy should be fitteed
    data : pd.DataFrame
        Data to be fitted
    monomer_a : str
        Name of the first monomer, for file naming
    monomer_b : str
        Name of the second monomer, for file naming
    orientation : str
        Name of the orientation, for file naming
    i : int
        Iteration number
    """

    # create a figure
    fig = plt.figure(figsize=(6, 5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    ax.plot(data["distance"], data["e_pair"], "o", ls="-", label="data", color="k")

    for j in range(len(filelist)):
        ax.plot(
            data["distance"][j],
            lj_dataset(params, j, filelist[j]),
            marker="x",
            ls="",
            color="r",
        )

    ax.set_xlabel(r"$d$ / $\mathrm{\AA}$")
    ax.set_ylabel(r"$E_\mathrm{LJ}$ / $\mathrm{m}E_\mathrm{h}$")

    # generate directory if it does not exist
    Path("./ljfit/img/").mkdir(parents=True, exist_ok=True)
    # remove the old plot files if they exist
    for file in Path("./ljfit/img/").glob(
        f"{monomer_b}-{monomer_a}-{orientation}_fit_{i}.pdf"
    ):
        file.unlink()

    fig.savefig(
        "./ljfit/img/"
        + monomer_b
        + "-"
        + monomer_a
        + "-"
        + orientation
        + f"_fit_{i}.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    # close figure after saving
    plt.close(fig)


def fit_lj_params(monomer_a: str, monomer_b: str, orientation: List[str]) -> None:
    """Fit Lennard-Jones parameters to a system of monomers, using the given orientations.

    Parameters
    ----------
    monomer_a : str
        The short name of the first monomer, used in the file names.
    monomer_b : str
        The short name of the second monomer, used in the file names.
    orientation : List[str]
        The orientation of the monomers to be considered, used in the file names.
    """

    # info
    print("Fitting Lennard-Jones parameters.\n")

    # file list of geometries
    ll_all = [
        get_file_list(
            "./" + monomer_b + "-" + monomer_a + "/",
            orientation[i] + "/[1-9]*/sapt.out",
        )
        for i in range(len(orientation))
    ]

    # file list of energies
    ff_all = [
        "./" + monomer_b + "-" + monomer_a + "/" + orientation[i] + "/e_summary.csv"
        for i in range(len(orientation))
    ]

    for i in range(len(orientation)):
        # get list of files
        geolist = ll_all[i]

        # read energy file
        df_energy = pd.read_csv(ff_all[i], sep=";")

        # drop the all rows with e_lj > 5
        # save the number of rows to a variable
        # remove as many rows from the beginning of geolist as there were rows dropped from the dataframe
        df_energy = df_energy[df_energy["e_pair"] < 5]
        while len(geolist) > len(df_energy):
            geolist.pop(0)

        # reshift index to begin at 0
        df_energy.reset_index(drop=True, inplace=True)

        # generate starting parameters for the fit
        start_params = get_params_to_fit(geolist)  # type: ignore

        # instantiate the Parameters object
        fit_params = Parameters()

        for j in range(len(geolist)):
            for _, row in start_params.iterrows():
                fit_params.add(
                    f"epsilon_{row['atom_pair']}_{j}",
                    value=0.85 * row["epsilon"],
                    # min=0.5 * row["epsilon"],
                    # max=0.9 * row["epsilon"],
                    vary=False,
                )
                fit_params.add(
                    f"sigma_{row['atom_pair']}_{j}",
                    value=0.85 * row["sigma"],
                    min=0.5 * row["sigma"],
                    max=0.9 * row["sigma"],
                    vary=True,
                )
                # make sure that the LJ params are consistent amont the different fit sets
                if j > 0:
                    fit_params[f"epsilon_{row['atom_pair']}_{j}"].expr = (
                        f"epsilon_{row['atom_pair']}_0"
                    )
                    fit_params[f"sigma_{row['atom_pair']}_{j}"].expr = (
                        f"sigma_{row['atom_pair']}_0"
                    )

        # info
        print(f"Starting the fit for {monomer_b}-{monomer_a}/{orientation[i]}.\n")
        print_lj_params(params_to_df(fit_params), 0)

        # fitting loop
        for c in range(10):
            # perform fit
            fit_out = minimize(
                get_residuals,
                fit_params,
                args=(geolist, df_energy),
                method="least_squares",
            )

            # info
            print_lj_params(params_to_df(fit_out.params), c + 1)  # type: ignore
            plot_fit(
                fit_out.params,
                geolist,
                df_energy,
                monomer_a,
                monomer_b,
                orientation[i],
                c + 1,
            )

            # update the parameters
            new_fit_params = Parameters()
            for j in range(len(geolist)):
                new_fit_params.add(
                    fit_out.params[f"epsilon_CG_B_{j}"].name,
                    value=fit_out.params[f"epsilon_CG_B_{j}"].value,
                    max=1.1 * fit_out.params[f"epsilon_CG_B_{j}"].value,
                    min=0.9 * fit_out.params[f"epsilon_CG_B_{j}"].value,
                    vary=c % 2 == 0,
                )
                new_fit_params.add(
                    fit_out.params[f"sigma_CG_B_{j}"].name,
                    value=fit_out.params[f"sigma_CG_B_{j}"].value,
                    max=1.1 * fit_out.params[f"sigma_CG_B_{j}"].value,
                    min=0.9 * fit_out.params[f"sigma_CG_B_{j}"].value,
                    vary=c % 2 == 1,
                )
                new_fit_params.add(
                    fit_out.params[f"epsilon_CG_FB_{j}"].name,
                    value=fit_out.params[f"epsilon_CG_FB_{j}"].value,
                    max=1.1 * fit_out.params[f"epsilon_CG_FB_{j}"].value,
                    min=0.9 * fit_out.params[f"epsilon_CG_FB_{j}"].value,
                    vary=c % 2 == 0,
                )
                new_fit_params.add(
                    fit_out.params[f"sigma_CG_FB_{j}"].name,
                    value=fit_out.params[f"sigma_CG_FB_{j}"].value,
                    max=1.1 * fit_out.params[f"sigma_CG_FB_{j}"].value,
                    min=0.9 * fit_out.params[f"sigma_CG_FB_{j}"].value,
                    vary=c % 2 == 1,
                )
                if j > 0:
                    new_fit_params[f"epsilon_CG_B_{j}"].expr = f"epsilon_CG_B_0"
                    new_fit_params[f"sigma_CG_B_{j}"].expr = f"sigma_CG_B_0"
                    new_fit_params[f"epsilon_CG_FB_{j}"].expr = f"epsilon_CG_FB_0"
                    new_fit_params[f"sigma_CG_FB_{j}"].expr = f"sigma_CG_FB_0"

            # calculate difference between the parameters
            df_diff = params_to_df(new_fit_params)
            df_diff[["epsilon", "sigma"]] = (
                df_diff[["epsilon", "sigma"]]
                - params_to_df(fit_params)[["epsilon", "sigma"]]
            )
            print("Difference between the parameters:")
            print(df_diff)

            # update the parameters
            fit_params = new_fit_params
