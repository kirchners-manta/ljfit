# Part of 'ljfit' package
"""
Calculate the multipole energy of a system using the multipole expansion.
"""

#############################################

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Tuple
from scipy.spatial import distance_matrix
from .. import __version__

# define constants
EH2KCAL = 627.503  # kcal/mol
ANGSTROM2BOHR = 1.8897259885789  # conversion factor from angstrom to bohr


def get_t_0(xyz_a: np.ndarray, xyz_b: np.ndarray) -> np.ndarray:
    """Calculate T_0 tensor of the multipole expansion.

    Parameters
    ----------
    xyz_a : np.ndarray
        coordinates of monomer A of shape (n_a, 3)
    xyz_b : np.ndarray
        coordinates of monomer B of shape (n_b, 3)

    Returns
    -------
    np.ndarray
        T_0 tensor.
        Has inverse distances between atoms as entries.
        Is matrix of shape (n_a, n_b).
    """

    # check if the input arrays have the correct shape
    if xyz_a.shape[1] != 3 or xyz_b.shape[1] != 3:
        raise ValueError("The input arrays have to have the shape (n, 3).")

    # calculate the distance matrix
    dist = distance_matrix(xyz_a, xyz_b)

    # create empty matrix
    t_0 = np.zeros(dist.shape)
    # loop over the matrix and get the inverse of the distances
    for i in range(t_0.shape[0]):
        for j in range(t_0.shape[1]):
            t_0[i, j] = 1 / dist[i, j]

    return t_0


def get_t_1(xyz_a: np.ndarray, xyz_b: np.ndarray) -> np.ndarray:
    """Calculate T_1 tensor of the multipole expansion.

    Parameters
    ----------
    xyz_a : np.ndarray
        coordinates of monomer A of shape (n_a, 3)
    xyz_b : np.ndarray
        coordinates of monomer B of shape (n_b, 3)

    Returns
    -------
    np.ndarray
        T_1 tensor.
        Has a shape of (n_a, n_b, 3).
    """

    # check if the input arrays have the correct shape
    if xyz_a.shape[1] != 3 or xyz_b.shape[1] != 3:
        raise ValueError("The input arrays have to have the shape (n, 3).")

    # calculate the distance matrix
    dist = distance_matrix(xyz_a, xyz_b)

    # create an empty 3d tensor
    t_1 = np.zeros((xyz_a.shape[0], xyz_b.shape[0], 3), dtype=float)
    # t_1(i,j,k) = -(x_j - x_i)_k / r_ij^3
    for i in range(xyz_a.shape[0]):
        for j in range(xyz_b.shape[0]):
            for k in range(3):
                t_1[i, j, k] = -(xyz_b[j, k] - xyz_a[i, k]) / dist[i, j] ** 3

    return t_1


def get_t_2(xyz_a: np.ndarray, xyz_b: np.ndarray) -> np.ndarray:
    """Calculate T_2 tensor of the multipole expansion.

    Parameters
    ----------
    xyz_a : np.ndarray
        coordinates of monomer A of shape (n_a, 3)
    xyz_b : np.ndarray
        coordinates of monomer B of shape (n_b, 3)

    Returns
    -------
    np.ndarray
        T_2 tensor.
        Has a shape of (n_a, n_b, 3, 3).
    """

    # check if the input arrays have the correct shape
    if xyz_a.shape[1] != 3 or xyz_b.shape[1] != 3:
        raise ValueError("The input arrays have to have the shape (n, 3).")

    # calculate the distance matrix
    dist = distance_matrix(xyz_a, xyz_b)

    # create an empty 3d tensor
    t_2 = np.zeros((xyz_a.shape[0], xyz_b.shape[0], 3, 3), dtype=float)
    # t_2(i,j,k,l) = 3 * (x_j - x_i)_k * (x_j - x_i)_l / r_ij^5 - delta_kl / r_ij^3
    for i in range(xyz_a.shape[0]):
        for j in range(xyz_b.shape[0]):
            for k in range(3):
                for l in range(3):
                    t_2[i, j, k, l] = (
                        3
                        * (xyz_b[j, k] - xyz_a[i, k])
                        * (xyz_b[j, l] - xyz_a[i, l])
                        / dist[i, j] ** 5
                    )
                    if k == l:
                        t_2[i, j, k, l] -= 1 / dist[i, j] ** 3
    return t_2


def get_t_thole(
    xyz_a: np.ndarray, xyz_b: np.ndarray, alpha: float = 1.139, damping: float = 1.507
) -> np.ndarray:
    """Calculate Thole's dipole field tensor.

    Parameters
    ----------
    xyz_a : np.ndarray
        coordinates of monomer A of shape (n_a, 3) in Angstrom
    xyz_b : np.ndarray
        coordinates of monomer B of shape (n_b, 3) in Angstrom
    alpha : float, optional
        Carbon (Graphene) polarizability, by default 1.139 in Angstrom^3
    damping : float, optional
        Thole damping parameter, by default 1.507 (dimensionless)

    Returns
    -------
    np.ndarray
        T_thole tensor.
        Has a shape of (n_a, n_b, 3, 3).
    """

    # check if the input arrays have the correct shape
    if xyz_a.shape[1] != 3 or xyz_b.shape[1] != 3:
        raise ValueError("The input arrays have to have the shape (n, 3).")

    # print(alpha, damping)

    # calculate the distance matrix
    dist = distance_matrix(xyz_a, xyz_b)

    # create an empty 3d tensor
    t_thole = np.zeros((xyz_a.shape[0], xyz_b.shape[0], 3, 3), dtype=float)

    # u_k = (x_j - x_i)_k / (alpha^2)^{1/6}
    # u = r_ij / (alpha^2)^{1/6}
    # t_thole(i,j,k,l) = 1/sqrt(alpha^2) * ( 3 * u_k * u_l / u^5 * [1 - ((a^3 * u^3) / 6 + (a^2 * u^2) / 2 + a * u + 1)) * exp(-a * u)]
    #                                    - delta_kl / u^3  * [1 - ((a^2 * u^2) / 2 + a * u + 1) * exp(-a * u))] )

    # calculate t2
    for i in range(xyz_a.shape[0]):
        for j in range(xyz_b.shape[0]):
            u = dist[i, j] / alpha ** (1 / 3)
            # only consider the distance if it is not zero
            if u == 0:
                continue
            # else calculate the tensor
            for k in range(3):
                for l in range(3):
                    t_thole[i, j, k, l] = (
                        1
                        / alpha
                        * (
                            3
                            * (xyz_b[j, k] - xyz_a[i, k])
                            * (xyz_b[j, l] - xyz_a[i, l])
                            * alpha ** (-2 / 3)
                            / u**5
                            * (
                                1
                                - (
                                    (damping**3 * u**3) / 6
                                    + (damping**2 * u**2) / 2
                                    + damping * u
                                    + 1
                                )
                            )
                            * np.exp(-damping * u)
                            - np.eye(3)[k, l]
                            / u**3
                            * (
                                1
                                - ((damping**2 * u**2) / 2 + damping * u + 1)
                                * np.exp(-damping * u)
                            )
                        )
                    )
    return t_thole


def get_e_multipole_0(
    charge_a: np.ndarray, charge_b: np.ndarray, xyz_a: np.ndarray, xyz_b: np.ndarray
) -> float:
    """Calculate the zeroth order multipole energy.

    Parameters
    ----------
    charge_a : np.ndarray
        Charges of monomer A in atomic units.
    charge_b : np.ndarray
        Charges of monomer B in atomic units.
    xyz_a : np.ndarray
        Cartesian coordinates of monomer A in Angstrom.
    xyz_b : np.ndarray
        Cartesian coordinates of monomer B in Angstrom.

    Returns
    -------
    float
        zeroth order multipole energy in Hartree.
    """
    # calculate the energy
    # e_0 = sum_i sum_j q_i * T_0(i,j) * q_j
    e_multipole_0 = np.sum(charge_a.reshape(-1, 1) * get_t_0(xyz_a, xyz_b) * charge_b)

    # correct the units:
    # charges are in atomic units already,
    # distances in Angstrom need to be converted to Bohr
    # prefactor 1/4pi*EPS_0 is in atomic units already
    e_multipole_0 /= ANGSTROM2BOHR

    return e_multipole_0


def get_e_multipole_1(
    charge_a: np.ndarray,
    charge_b: np.ndarray,
    dipole_a: np.ndarray,
    dipole_b: np.ndarray,
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
) -> float:
    """Calculate the first order multipole energy.

    Parameters
    ----------
    charge_a : np.ndarray
        Charges of monomer A in atomic units.
    charge_b : np.ndarray
        Charges of monomer B in atomic units.
    dipole_a : np.ndarray
        Dipole moments of monomer A in atomic units.
    dipole_b : np.ndarray
        Dipole moments of monomer B in atomic units.
    xyz_a : np.ndarray
        Cartesian coordinates of monomer A in Angstrom.
    xyz_b : np.ndarray
        Cartesian coordinates of monomer B in Angstrom.

    Returns
    -------
    float
        first order multipole energy in Hartree.
    """

    # calculate the energy
    # e_1 = sum_i sum_j sum_k [q_i * T_1(i,j,k) * mu_jk - mu_ik * T_1(i,j,k) * q_j]
    e_multipole_1 = np.sum(
        np.einsum("i,ijk->jk", charge_a, get_t_1(xyz_a, xyz_b)) * dipole_b
    ) - np.sum(np.einsum("ik,ijk->j", dipole_a, get_t_1(xyz_a, xyz_b)) * charge_b)

    # correct the units:
    # charges are in atomic units already,
    # distances in Angstrom need to be converted to Bohr
    # dipole moments are in atomic units already
    # prefactor 1/4pi*EPS_0 is in atomic units already
    e_multipole_1 /= ANGSTROM2BOHR**2

    return e_multipole_1


def get_e_multipole_2(
    charge_a: np.ndarray,
    charge_b: np.ndarray,
    dipole_a: np.ndarray,
    dipole_b: np.ndarray,
    quadrupole_a: np.ndarray,
    quadrupole_b: np.ndarray,
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
) -> float:
    """Calculate the second order multipole energy.

    Parameters
    ----------
    charge_a : np.ndarray
        Charges of monomer A in atomic units.
    charge_b : np.ndarray
        Charges of monomer B in atomic units.
    dipole_a : np.ndarray
        Dipole moments of monomer A in atomic units.
    dipole_b : np.ndarray
        Dipole moments of monomer B in atomic units.
    quadrupole_a : np.ndarray
        Quadrupole moments of monomer A in atomic units.
    quadrupole_b : np.ndarray
        Quadrupole moments of monomer B in atomic units.
    xyz_a : np.ndarray
        Cartesian coordinates of monomer A in Angstrom.
    xyz_b : np.ndarray
        Cartesian coordinates of monomer B in Angstrom.

    Returns
    -------
    float
        second order multipole energy in Hartree.
    """

    # calculate the energy
    # e_2 = sum_i sum_j sum_k sum_l [q_i * T_2(i,j,k,l) * Q_jkl / 3 - mu_ik * T_2(i,j,k,l) * mu_jl + Q_ikl * T_2(i,j,k,l) * q_j / 3]
    e_multipole_2 = (
        np.sum(
            np.einsum("i,ijkl->jkl", charge_a, get_t_2(xyz_a, xyz_b)) * quadrupole_b / 3
        )
        - np.sum(np.einsum("ik,ijkl->jl", dipole_a, get_t_2(xyz_a, xyz_b)) * dipole_b)
        + np.sum(
            np.einsum("ikl,ijkl->j", quadrupole_a, get_t_2(xyz_a, xyz_b)) * charge_b / 3
        )
    )

    # correct the units:
    # charges are in atomic units already,
    # distances in Angstrom need to be converted to Bohr
    # dipole moments are in atomic units already
    # quadrupole moments are in atomic
    # prefactor 1/4pi*EPS_0 is in atomic units already
    e_multipole_2 /= ANGSTROM2BOHR**3

    return e_multipole_2


def get_e_multipole_truncation(
    charge_a: np.ndarray,
    charge_b: np.ndarray,
    dipole_b: np.ndarray,
    quadrupole_b: np.ndarray,
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
) -> float:
    """Calculate the multipole energy with truncation.
    Using charges only for monomer A and charges, dipole and quadrupole moments for monomer B.

    Parameters
    ----------
    charge_a : np.ndarray
        Charges of monomer A in atomic units.
    charge_b : np.ndarray
        Charges of monomer B in atomic units.
    dipole_b : np.ndarray
        Dipole moments of monomer B in atomic units.
    quadrupole_b : np.ndarray
        Quadrupole moments of monomer B in atomic units.
    xyz_a : np.ndarray
        Cartesian coordinates of monomer A in Angstrom.
    xyz_b : np.ndarray
        Cartesian coordinates of monomer B in Angstrom.

    Returns
    -------
    float
        multipole energy with truncation in Hartree.
    """

    # calculate the energy
    # e_trunc = sum_i sum_j [q_i * T_0(i,j) * q_j + sum_k [q_i * T_1(i,j,k) * mu_jk] + sum_k sum_l [q_i * T_2(i,j,k,l) * Q_jkl / 3]]
    e_trunc = (
        get_e_multipole_0(charge_a, charge_b, xyz_a, xyz_b)
        + get_e_multipole_1(
            charge_a, charge_b, np.zeros((charge_a.shape[0], 3)), dipole_b, xyz_a, xyz_b
        )
        + get_e_multipole_2(
            charge_a,
            charge_b,
            np.zeros((charge_a.shape[0], 3)),
            dipole_b,
            np.zeros((charge_a.shape[0], 3, 3)),
            quadrupole_b,
            xyz_a,
            xyz_b,
        )
    )

    return e_trunc


def get_electric_field(
    xyz_a: np.ndarray, xyz_b: np.ndarray, charges_a: np.ndarray
) -> np.ndarray:
    """Calculate the electric field that monomer A exerts on monomer B.

    Parameters
    ----------
    xyz_a : np.ndarray
        Cartesian coordinates of monomer A in Angstrom. Has a shape of (n_a, 3).
    xyz_b : np.ndarray
        Cartesian coordinates of monomer B in Angstrom. Has a shape of (n_b, 3).
    charge_a : np.ndarray
        Charges of molecule A in atomic units.

    Returns
    -------
    np.ndarray
        Electric field that monomer A exerts on monomer B. Has a shape of (nb, 3). In a.u. / Angstrom^2.
    """
    # check if the input arrays have the correct shape
    if xyz_a.shape[1] != 3 or xyz_b.shape[1] != 3:
        raise ValueError("The input arrays have to have the shape (n, 3).")

    # calculate the distance matrix
    dist = distance_matrix(xyz_a, xyz_b)

    # create an empty 2d tensor
    e_field = np.zeros((xyz_b.shape[0], 3), dtype=float)
    # e_field(j,k) = sum_i q_i * (x_j - x_i)_k / r_ij^3
    for j in range(xyz_b.shape[0]):
        for k in range(3):
            e_field[j, k] = np.sum(
                charges_a * (xyz_b[j, k] - xyz_a[:, k]) / dist[:, j] ** 3
            )

    return e_field


def dipoles_matrix_inversion(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    charges_a: np.ndarray,
    alpha: float = 1.139,
    damping: float = 1.507,
) -> np.ndarray:
    """Self-consistently solve the induced dipoles in monomer B.

    Parameters
    ----------
    xyz_a : np.ndarray
        Cartesian coordinates of monomer A in Angstrom.
    xyz_b : np.ndarray
        Cartesian coordinates of monomer B in Angstrom.
    charges_a : np.ndarray
        Charges of monomer A in atomic units.
    alpha : float, optional
        Carbon (Graphene) polarizability, by default 1.139 in Angstrom^3
    damping : float, optional
        Thole damping parameter, by default 1.507 (dimensionless)

    Returns
    -------
    np.ndarray
        Induced dipoles in monomer B in a.u. * Angstrom.
    """

    # the following formula is used to solve the induced dipoles
    # mu_i = akpha_C * [ sum_j(E_j) + sum_k(not i) T_thole(i,k) * mu_k ]
    # this is transformed to some matrix equation
    # R * M = E
    # M = R^-1 * E
    # E is the electric field that molecule A exerts on atom B, a matrix of shape (n_b, 3)
    # M is the induced dipoles, a matrix of shape (n_b, 3)
    # R is the relay matrix, a matrix of shape (n_b, n_b)
    # R has alpha_C ^ -1 on the diagonal and T_thole on the off-diagonal elements
    # and then, R^-1 = (alpha_C^-1 - T_thole)^-1

    # start
    # calculate the electric field
    e_field = get_electric_field(xyz_a, xyz_b, charges_a)

    # calculate the relay matrix
    # R = alpha_C^-1 - T_thole
    # get thole tensor (dimensions: n_b, n_b, 3, 3) and transform it to a 2D matrix (dimensions: n_b, n_b)
    t_thole = get_t_thole(xyz_b, xyz_b, alpha=alpha, damping=damping)
    t_thole_2d = np.zeros((xyz_b.shape[0], xyz_b.shape[0]))
    for i in range(xyz_b.shape[0]):
        for j in range(xyz_b.shape[0]):
            t_thole_2d[i, j] = np.trace(t_thole[i, j])
    # calculate the inverted relay matrix
    r = np.linalg.inv(np.linalg.inv(np.eye(xyz_b.shape[0]) * alpha) - t_thole_2d)

    # check if the inverted relay matrix is symmetric
    if not np.allclose(r, r.T):
        raise ValueError("The inverted relay matrix is not symmetric.")

    # solve the equation
    dipoles = np.dot(r, e_field)

    # units are currently in a.u. * Angstrom

    return dipoles


def get_e_pol(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    charges_a: np.ndarray,
    alpha: float = 1.139,
    damping: float = 1.507,
) -> float:
    """Calculate the polarization energy.

    Parameters
    ----------
    xyz_a : np.ndarray
        Cartesian coordinates of monomer A in Angstrom.
    xyz_b : np.ndarray
        Cartesian coordinates of monomer B in Angstrom.
    charges_a : np.ndarray
        Charges of monomer A in atomic units.
    alpha : float, optional
        Carbon (Graphene) polarizability, by default 1.139 in Angstrom^3
    damping : float, optional
        Thole damping parameter, by default 1.507 (dimensionless)

    Returns
    -------
    float
        polarization energy in Hartree.
    """

    # e_pol = -1/2 * sum_i mu_i * E_i
    # where E_i is the electric field that molecule A exerts on atom B
    # and mu_i is the induced dipole of atom i
    e_pol = -0.5 * np.sum(
        np.dot(
            dipoles_matrix_inversion(xyz_a, xyz_b, charges_a, alpha, damping).T,
            get_electric_field(xyz_a, xyz_b, charges_a),
        )
    )

    # correct the units:
    # charges are in atomic units already,
    # distances in Angstrom need to be converted to Bohr
    # dipole moments are in a.u. * Angstrom
    # electric field is in a.u. / Angstrom^2
    # prefactor 1/4pi*EPS_0 is in atomic units already
    e_pol *= ANGSTROM2BOHR

    return e_pol
