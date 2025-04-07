# LJFit
---

This repo is to be understood in conjunction with the [Supplementary Information repository](https://github.com/kirchners-manta/il-graphene-ff) for our paper on an ionic liquid/graphite interface.
Take a look there for further information. 

The software in this repository can be installed in the command line using `pip install .` in the root directory of the repository.
If used with the files provided in the repository mentioned above, the software can be used to fit Lennard-Jones parameters for ions at a graphite interface.
Therefore, first, the energies have to be extracted from the output files of the SAPT calculations (see pseudocode 1 below, `ljfit -e`).
Then, as a second step, the Lennard-Jones parameters have to be fitted to the pairwise energies (see pseudocode 2 below, `ljfit -f`).

Pseudocode example of the workflow implemented here:

```python
# iterate over all ion orientations
for orientation in all_ion_orientations:
    # read multipole moments from GDMA and CHELPG charges from ORCA
    multipoles = read_multipole_moments(orientation)
    chelpg_charges = read_chelpg_charges(orientation)
    # iterate over all distances between the ion and the surface
    # for which an SAPT calculation was performed
    for job in sapt_calculations:
        # process SAPT output
        # get coordinates of monomers, energies, etc.
        xyz, e_sapt0, ... = proc_sapt_output(job)
        # calculate multipole energy
        e_mult = calc_e_mult(xyz, multipoles, chelpg_charges)
        # calculate classical polarization energy
        e_pol = calc_e_pol(xyz, chelpg_charges)
        # calculate interaction energy
        e_int = e_sapt0 - e_mult
        # calculate pairwise energy
        e_pair = e_int - e_pol
        # store results
        store_results(job, all_energy_terms)
```

```python
# iterate over all ion orientations
for orientation in all_ion_orientations:
    # generate initial guess for the LJ parameters from CL&Pol
    # the parameter object contains information 
    # about which parameters are fixed and which are optimized
    lj_params = generate_lj_params(orientation)
    # read pairwise energies
    e_pair = read_pairwise_energies(orientation)
    # initialize convergence flag
    converged = [False] * len(lj_params)

    while not all(converged):
        # save current LJ parameters
        lj_params_old.append(lj_params)
        # optimize LJ parameters
        lj_params = optimize_lj_params(lj_params, e_pair, converged)
        # check convergence
        converged = check_convergence(lj_params, lj_params_old)
        if not all(converged):
            # update which parameters to optimize
            lj_params = update_params(lj_params, lj_params_old, converged)
        else:
            # print results
            print_results(lj_params)
```