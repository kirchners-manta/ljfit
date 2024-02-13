# Part of 'ljfit' package
"""
Fit Lennard-Jones parameters to a system of monomers.
"""

#############################################

from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Parameters, minimize
from scipy.spatial import distance_matrix

from .. import __version__
from ..energy import get_xyz_from_sapt
from ..helpers import (
    get_file_list,
    params_to_df,
    print_lj_params,
    custom_print,
    write_params_to_csv,
)

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


def get_residuals(
    params: Parameters, filelist: List[Path], data: pd.Series
) -> List[float]:
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


def get_start_params(filelist: List[str | Path]) -> Tuple[Parameters, List[str]]:
    """Generate a DataFrame with initial Lennard-Jones parameters for a fit.

    Parameters
    ----------
    filelist : List[str  |  Path]
        List of file paths to the SAPT input/ output files with the geometries.
        Only the first file is used to generate the parameters.

    Returns
    -------
    Parameters
        Initial parameters for the fit.
    List[str]
        List of atom pairs.
    """

    # LJ parameters, serving as start points for fit.
    # epsilon values taken based on Ref: 10.1039/c8cp01677a
    # the polarizabilities from table 3 are taken, rounded and divided by 4 as the epsilon values
    epsilon_start = {  # 'atom_A-atom_B': epsilon in kcal/mol
        "CG-B": 0.1445,  # C(Graphene) - B(BF4-)
        "CG-FB": 0.1745,  # C(Graphene) - F(BF4-)
        "CG-NA": 0.2800,  # C(Graphene) - N(Imidazolium)
        "CG-CR": 0.2625,  # C(Graphene) - CR(Imidazolium)
        "CG-CW": 0.2625,  # C(Graphene) - CW(Imidazolium)
        "CG-C1": 0.26,  # C(Graphene) - C1(Imidazolium)
        "CG-HCR": 0.07,  # C(Graphene) - HCR(Imidazolium)
        "CG-HCW": 0.07,  # C(Graphene) - HCW(Imidazolium)
        "CG-HC": 0.07,  # C(Graphene) - HC(Imidazolium side chain)
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

    sigma_start = {  # 'atom_A-atom_B': sigma in Angstrom
        "CG-B": 3.0,  # 3.565,  # C(Graphene) - B(BF4-)
        "CG-FB": 3.2,  # 3.335,  # C(Graphene) - F(BF4-)
        "CG-NA": 3.20,  # C(Graphene) - N(Imidazolium)
        "CG-CR": 3.3,  # C(Graphene) - CR(Imidazolium)
        "CG-CW": 3.3,  # C(Graphene) - CW(Imidazolium)
        "CG-C1": 3.5,  # C(Graphene) - C1(Imidazolium)
        "CG-HCR": 2.7,  # C(Graphene) - HCR(Imidazolium)
        "CG-HCW": 2.7,  # C(Graphene) - HCW(Imidazolium)
        "CG-HC": 2.9,  # C(Graphene) - HC(Imidazolium side chain)
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
    atom_pairs_tuple = list(set(itertools.product(b["atom"], a["atom"])))
    # sort the atom pairs in ascending order with respect to their epsilon
    atom_pairs_tuple = sorted(
        atom_pairs_tuple,
        key=lambda x: (
            epsilon_start[f"{x[0]}-{x[1]}"],
            sigma_start[f"{x[0]}-{x[1]}"],
        ),
    )
    # join the atom names with a _
    atom_pairs = [f"{pair[0]}_{pair[1]}" for pair in atom_pairs_tuple]

    # instantiate Parameters object
    start_params = Parameters()
    # iterate through the files and atom pairs and assign the initial LJ parameters
    for i, _ in enumerate(filelist):
        for j, pair in enumerate(atom_pairs_tuple):
            start_params.add(
                f"epsilon_{atom_pairs[j]}_{i}",
                value=epsilon_start[f"{pair[0]}-{pair[1]}"] / EH2KCAL,
                max=1.1 * epsilon_start[f"{pair[0]}-{pair[1]}"] / EH2KCAL,
                vary=(j == 0),
            )
            start_params.add(
                f"sigma_{atom_pairs[j]}_{i}",
                value=sigma_start[f"{pair[0]}-{pair[1]}"] * ANGSTROM2BOHR,
                max=1.1 * sigma_start[f"{pair[0]}-{pair[1]}"] * ANGSTROM2BOHR,
                vary=(j == 0),
            )
            # make sure that the LJ params are consistent amont the different fit sets
            if i > 0:
                start_params[f"epsilon_{atom_pairs[j]}_{i}"].expr = (
                    f"epsilon_{atom_pairs[j]}_0"
                )
                start_params[f"sigma_{atom_pairs[j]}_{i}"].expr = (
                    f"sigma_{atom_pairs[j]}_0"
                )
            # make sure that the order of the epsilon values stays the same
            if j > 0:
                start_params[f"epsilon_{atom_pairs[j]}_{i}"].min = start_params[
                    f"epsilon_{atom_pairs[j-1]}_{i}"
                ].value
            else:
                start_params[f"epsilon_{atom_pairs[j]}_{i}"].min = (
                    start_params[f"epsilon_{atom_pairs[j]}_{i}"].value * 0.9
                )

    return [start_params, atom_pairs]


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


def scaling(kind: str, param: str, i: int) -> float:
    """Scaling function for the Lennard-Jones parameters.

    Parameters
    ----------
    kind : str
        Kind of scaling, either "max" or "min".
    param : str
        Parameter to be scaled, either "epsilon" or "sigma".
    i: int
        Iteration number.

    Returns
    -------
    float
        Scaling factor.
    """

    if kind == "max":
        if param == "epsilon":
            if i < 20:
                return 1.1
            else:
                return 1.05
        elif param == "sigma":
            if i < 20:
                return 1.1
            else:
                return 1.05
    elif kind == "min":
        if param == "epsilon":
            if i < 20:
                return 0.9
            else:
                return 0.95
        elif param == "sigma":
            if i < 20:
                return 0.9
            else:
                return 0.95
    else:
        return 0.0

    return 0.0


def update_params(fit_result: Parameters, atom_pairs: List[str], i: int) -> Parameters:
    """Update the parameters of the fit.

    Parameters
    ----------
    fit_result : Parameters
        Parameters object with the result of the fit.
        Used to update generate parameters.
    atom_pairs:
        List of atom pairs.
    i : int
        Iteration number.

    Returns
    -------
    Parameters
        Updated parameters, used for the next iteration.
    """
    # number of independent parameters
    n_params = len(atom_pairs)
    # number of parameter sets
    n_sets = len(fit_result) // (n_params * 2)
    # instantiate the new Parameters object
    new_params = Parameters()

    # generate the new parameters
    for j in range(n_sets):
        for k in range(n_params):
            if k == 0:
                # first epsilon is set, the other epsilons are dependent on the first one
                new_params.add(
                    f"epsilon_{atom_pairs[k]}_{j}",
                    value=fit_result[f"epsilon_{atom_pairs[k]}_{j}"].value,
                    min=fit_result[f"epsilon_{atom_pairs[k]}_{j}"].value
                    * scaling("min", "epsilon", i),
                    max=fit_result[f"epsilon_{atom_pairs[k]}_{j}"].value
                    * scaling("max", "epsilon", i),
                    vary=(k == i or i % n_params == k),
                )
            else:
                # all other epsilons are dependent on the first one,
                # the parameters were previosly sorted in ascending order
                # with respect to their polarizability
                new_params.add(
                    f"epsilon_{atom_pairs[k]}_{j}",
                    value=fit_result[f"epsilon_{atom_pairs[k]}_{j}"].value,
                    min=fit_result[f"epsilon_{atom_pairs[k-1]}_{j}"].value,
                    max=fit_result[f"epsilon_{atom_pairs[k]}_{j}"].value
                    * scaling("max", "epsilon", i),
                    vary=(k == i or i % n_params == k),
                )

            new_params.add(
                f"sigma_{atom_pairs[k]}_{j}",
                value=fit_result[f"sigma_{atom_pairs[k]}_{j}"].value,
                min=fit_result[f"sigma_{atom_pairs[k]}_{j}"].value
                * scaling("min", "sigma", i),
                max=fit_result[f"sigma_{atom_pairs[k]}_{j}"].value
                * scaling("max", "sigma", i),
                vary=(k == i or i % n_params == k),
            )
            # make sure that the LJ params are consistent amont the different fit sets
            if j > 0:
                new_params[f"epsilon_{atom_pairs[k]}_{j}"].expr = (
                    f"epsilon_{atom_pairs[k]}_0"
                )
                new_params[f"sigma_{atom_pairs[k]}_{j}"].expr = (
                    f"sigma_{atom_pairs[k]}_0"
                )
        # eps(CG_B) must be > eps(CG_FB)
        new_params[f"epsilon_CG_B_{j}"].max = new_params[f"epsilon_CG_FB_{j}"].value

    return new_params


def check_param_convergence(
    fit_params: Parameters,
    d_params: Parameters,
    dd_params: Parameters,
    l_params_converged: List[bool],
    tol_epsilon: float = 5e-4,
    tol_sigma: float = 5e-3,
) -> bool:
    """Check if the parameters have converged.

    Parameters
    ----------
    fit_params : Parameters
        Parameters object with the result of the fit.
    d_params : Parameters
        Parameters object with the previous parameters.
    dd_params : Parameters
        Parameters object with the parameters before the previous ones.
    l_params_converged : List[bool]
        List of bools, indicating which of the parameters are converged.
    tol_epsilon : float
        Tolerance for the epsilon parameter in a.u.
    tol_sigma : float
        Tolerance for the sigma parameter in a.u.

    Returns
    -------
    bool
        True if the entire fit has converged, False otherwise.
    """

    # if we have reached final conversion and no parameter is varied, return True
    if d_params == fit_params:
        return True
    else:
        for k, key in enumerate(fit_params):
            # scan only relevant parameters
            if k < len(l_params_converged):
                if key.startswith("epsilon_"):
                    if (
                        abs(fit_params[key].value - dd_params[key].value)
                        < tol_epsilon / EH2KCAL
                    ):
                        fit_params[key].vary = False
                        l_params_converged[k] = True
                elif key.startswith("sigma_"):
                    if (
                        abs(fit_params[key].value - dd_params[key].value)
                        < tol_sigma * ANGSTROM2BOHR
                    ):
                        fit_params[key].vary = False
                        l_params_converged[k] = True

            # if no parameter is varied, check if all parameters are converged
            # if yes, return True
            # if no, continue with the next iteration, return False
            if all([not fit_params[key].vary for key in fit_params]):
                if all(l_params_converged):
                    return True
                else:
                    # find the first parameter that is not converged and set it to vary
                    for j, conv in enumerate(l_params_converged):
                        if not conv:
                            fit_params[list(fit_params.keys())[j]].vary = True
                            return False
    return False


def fit_lj_params(
    monomer_a: str, monomer_b: str, orientation: List[str], print_level: int
) -> None:
    """Fit Lennard-Jones parameters to a system of monomers, using the given orientations.

    Parameters
    ----------
    monomer_a : str
        The short name of the first monomer, used in the file names.
    monomer_b : str
        The short name of the second monomer, used in the file names.
    orientation : List[str]
        The orientation of the monomers to be considered, used in the file names.
    print_level : int
        The level of verbosity for the output.
    """

    # info
    custom_print("The following systems will be fitted:", 2, print_level)
    for i in range(len(orientation)):
        custom_print(f"{monomer_b}-{monomer_a}/{orientation[i]}", 2, print_level)
    custom_print("", 2, print_level)

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

        # get list of files corresponding to the current orientation
        geolist = ll_all[i]
        # read energy file
        df_energy = pd.read_csv(ff_all[i], sep=";")
        # drop the all rows with e_lj > 5
        # remove as many rows from the beginning of geolist as there were rows dropped from the dataframe
        df_energy = df_energy[df_energy["e_pair"] < 5]
        while len(geolist) > len(df_energy):
            geolist.pop(0)
        # reshift index to begin at 0
        df_energy.reset_index(drop=True, inplace=True)

        # generate starting parameters for the fit
        fit_params, atom_pairs = get_start_params(geolist)  # type: ignore

        # initialize previous values of parameters
        d_params = fit_params

        # generate a list with bools, indicating if the parameters are converged
        # the list has length of the number of parameters and follows the enumeration of the fit_params object
        # initially, all parameters are not converged
        l_params_converged = [False for _ in range(len(fit_params) // len(geolist))]

        # info
        custom_print(
            f"Starting the fit for {monomer_b}-{monomer_a}/{orientation[i]}.\n",
            1,
            print_level,
        )
        custom_print(print_lj_params(params_to_df(fit_params), 0), 2, print_level)

        # fitting loop
        converged = False
        c = 0
        while not converged:

            # perform fit
            fit_out = minimize(
                get_residuals,
                fit_params,
                args=(geolist, df_energy),
                method="least_squares",
            )
            c += 1

            # info
            custom_print(print_lj_params(params_to_df(fit_out.params), c), 2, print_level)  # type: ignore

            # generate the new parameters and save the old ones
            dd_params = d_params
            d_params = fit_params
            fit_params = update_params(fit_out.params, atom_pairs, c)  # type: ignore

            # check for parameter convergence
            if c > 2:
                converged = check_param_convergence(
                    fit_params, d_params, dd_params, l_params_converged
                )

            # info
            if print_level > 0:
                if (print_level == 2 and c % 5 == 0) or print_level == 3:
                    write_params_to_csv(params_to_df(fit_params), orientation[i], c)
                    plot_fit(
                        fit_out.params,  # type: ignore
                        geolist,
                        df_energy,
                        monomer_a,
                        monomer_b,
                        orientation[i],
                        c,
                    )

        # info
        custom_print("*** Fit converged ***\n", 1, print_level)
        custom_print(print_lj_params(params_to_df(fit_params), -1), 0, print_level)
        write_params_to_csv(params_to_df(fit_params), orientation[i], -1)
        plot_fit(
            fit_out.params,  # type: ignore
            geolist,
            df_energy,
            monomer_a,
            monomer_b,
            orientation[i],
            -1,
        )
